import io
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

st.set_page_config(page_title="Forecast Nov-Dic 2026", layout="wide")
st.markdown("## Forecast de ventas | Noviembre y diciembre 2026")
st.caption("Histórico mensual comparable y pronóstico conectado, con filtros de categoría, combo y producto.")

with st.expander("¿Cómo funciona el modelo?", expanded=False):
    st.markdown(
        """
### Resumen

El forecast combina **tres modelos globales CatBoost** y una corrección estacional controlada. El objetivo es estimar las ventas de noviembre y diciembre de 2026 sin limitarse a copiar el cierre de 2025.

### 1. Información utilizada

El modelo aprende con datos mensuales de 2024, 2025 y 2026 hasta octubre:

- Venta monetaria.
- Unidades vendidas.
- Precio promedio, recalculado como `VENTA / UNIDADES`.
- Código, categoría, familia, tipo de producto y condición de combo.
- Rezagos de 1, 2, 3, 6, 12 y 24 meses.
- Promedios y variabilidad de los últimos 3, 6 y 12 meses.
- Mes del año y tendencia reciente.

### 2. Tres pronósticos CatBoost

Se entrenan tres modelos:

1. **Venta directa:** pronostica el importe monetario.
2. **Unidades:** pronostica cuántas unidades se venderán.
3. **Precio promedio:** pronostica el precio medio esperado.

La venta estructural se calcula así:

`Venta estructural = Unidades pronosticadas × Precio pronosticado`

La predicción base combina:

`Predicción base = 85% venta directa + 15% venta estructural`

### 3. Estacionalidad de noviembre y diciembre

El ajuste externo solo se activa cuando el SKU presenta un índice estacional mínimo de **1.20**. Esto significa que históricamente noviembre y diciembre deben vender al menos 20% más que el resto del año.

Cuando se activa, se construye una referencia con:

- 65% del mismo mes más reciente.
- 35% del mismo mes de hace dos años.
- Crecimiento enero-octubre del año objetivo frente al año anterior.

La señal queda combinada como:

`Predicción estacional = 85% predicción base + 15% referencia histórica YTD`

### 4. Piso parcial validado

La venta del mismo mes anterior no se usa como piso rígido. Solo funciona como protección ante una subestimación extrema.

- En **noviembre** se recupera 45% de la distancia hacia el piso parcial.
- En **diciembre** se recupera 60%, porque diciembre mostró mayor subestimación en el backtest.
- Si la predicción ya está por encima del piso, no se agrega corrección.

Esto permite que el forecast quede por encima o por debajo de 2025 según la tendencia, sin copiar exactamente ese valor.

### 5. Validación histórica

La configuración se validó entrenando hasta octubre de 2025 y pronosticando noviembre-diciembre de 2025. Sin combos, los resultados fueron:

| Métrica | Resultado |
|---|---:|
| WAPE | 18.42% |
| MAE | $39,321.84 |
| RMSE | $113,103.17 |
| R² | 0.9457 |
| Sesgo | -6.43% |

El sesgo negativo indica que el modelo sigue siendo conservador. Las métricas son una referencia histórica, no una garantía del error real de 2026.

### 6. Interpretación de la gráfica

- **Línea azul:** ventas reales hasta octubre de 2026.
- **Línea naranja punteada:** pronóstico de noviembre y diciembre de 2026.
- Los filtros permiten analizar categoría, productos sin combos, combos o un SKU individual.
- El histórico utiliza el mismo universo de SKU activos incluidos en el forecast.
"""
    )

with st.expander("Fórmulas principales", expanded=False):
    st.code(
        """Venta estructural = Unidades predichas * Precio predicho

Predicción base = 0.85 * Venta directa + 0.15 * Venta estructural

Predicción estacional = 0.85 * Predicción base + 0.15 * Referencia histórica YTD

Si Predicción combinada < Piso parcial:
    Noviembre: Predicción final = combinada + 0.45 * (piso - combinada)
    Diciembre: Predicción final = combinada + 0.60 * (piso - combinada)
Si no:
    Predicción final = Predicción combinada""",
        language="text",
    )

MESES = {
    "ENERO": 1, "FEBRERO": 2, "MARZO": 3, "ABRIL": 4,
    "MAYO": 5, "JUNIO": 6, "JULIO": 7, "AGOSTO": 8,
    "SEPTIEMBRE": 9, "OCTUBRE": 10, "NOVIEMBRE": 11,
    "DICIEMBRE": 12,
}

CAT_FEATURES = ["CODIGO", "CATEGORIA", "FAMILIA", "TIPO", "COMBOS"]
LAGS = [1, 2, 3, 6, 12, 24]
NUM_FEATURES = (
    [f"lag{x}" for x in LAGS]
    + [
        "mean3", "std3", "mean6", "std6", "mean12", "std12",
        "sin_mes", "cos_mes", "MES_NUM", "ANO", "tendencia_3_12",
    ]
)
FEATURES = NUM_FEATURES + CAT_FEATURES

# Configuración seleccionada en el backtest Nov-Dic 2025.
PESO_VENTA_DIRECTA = 0.85
PESO_ESTRUCTURAL = 0.15
PESO_FORECAST_ESTACIONAL = 0.85
UMBRAL_ESTACIONAL = 1.20


def normalizar_codigo(serie):
    return serie.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def _normalizar_encabezado(valor):
    """Normaliza encabezados de Excel sin depender de espacios, acentos o saltos."""
    import re
    import unicodedata
    texto = unicodedata.normalize("NFKD", str(valor))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.replace("\n", " ").replace("\r", " ").strip().upper()
    texto = re.sub(r"\s+", " ", texto)
    return texto


def _buscar_columna(columnas, aliases, prefijos=None):
    """Busca encabezados exactos y también variantes como UNIDADES 2025."""
    mapa = {_normalizar_encabezado(c): c for c in columnas}
    for alias in aliases:
        alias_normalizado = _normalizar_encabezado(alias)
        if alias_normalizado in mapa:
            return mapa[alias_normalizado]
    for prefijo in prefijos or []:
        prefijo_normalizado = _normalizar_encabezado(prefijo)
        for encabezado_normalizado, encabezado_original in mapa.items():
            if encabezado_normalizado.startswith(prefijo_normalizado):
                return encabezado_original
    return None


def _inferir_ano(nombre_archivo):
    import re
    coincidencia = re.search(r"(?:19|20)\d{2}", str(nombre_archivo))
    return int(coincidencia.group()) if coincidencia else None


def _inferir_mes(nombre_hoja):
    nombre = _normalizar_encabezado(nombre_hoja)
    for mes, numero in MESES.items():
        if mes in nombre:
            return numero
    return None


def cargar_archivos(archivos):
    """Carga robusta de los tres Excel.

    El año se toma del nombre VENTAS 2024/2025/2026. La función acepta una hoja
    por mes o una tabla consolidada, y reconoce variaciones de encabezados.
    No usa caché para evitar reutilizar una lectura anterior incompleta.
    """
    partes = []
    diagnostico = []

    aliases = {
        "CODIGO": ["CODIGO", "CÓDIGO", "SKU", "COD PRODUCTO", "COD. PRODUCTO"],
        "VENTA": ["VENTA", "VENTAS", "MONTO VENTA", "VENTA NETA"],
        "UNIDADES": ["UNIDADES", "UNIDAD", "UNIDADES VENDIDAS", "CANTIDAD", "PIEZAS"],
        "MES": ["MES", "NOMBRE MES"],
        "DESCRIPCION": ["DESCRIPCION", "DESCRIPCIÓN", "PRODUCTO", "NOMBRE PRODUCTO"],
        "CATEGORIA": ["CATEGORIA", "CATEGORÍA"],
        "FAMILIA": ["FAMILIA"],
        "TIPO": ["TIPO"],
        "COMBOS": ["COMBOS", "COMBO", "ES COMBO"],
    }

    for archivo in archivos:
        nombre = archivo.name
        ano_archivo = _inferir_ano(nombre)
        if ano_archivo not in {2024, 2025, 2026}:
            diagnostico.append({
                "Archivo": nombre, "Hoja": "-", "Estado": "ERROR",
                "Filas": 0, "Año asignado": ano_archivo,
                "Detalle": "El nombre debe contener 2024, 2025 o 2026",
            })
            continue

        try:
            contenido = archivo.getvalue()
            libro = pd.ExcelFile(io.BytesIO(contenido), engine="openpyxl")
        except Exception as exc:
            diagnostico.append({
                "Archivo": nombre, "Hoja": "-", "Estado": "ERROR",
                "Filas": 0, "Año asignado": ano_archivo,
                "Detalle": f"No fue posible abrir el Excel: {exc}",
            })
            continue

        for hoja in libro.sheet_names:
            try:
                h = pd.read_excel(libro, sheet_name=hoja)
            except Exception as exc:
                diagnostico.append({
                    "Archivo": nombre, "Hoja": hoja, "Estado": "ERROR",
                    "Filas": 0, "Año asignado": ano_archivo,
                    "Detalle": f"No fue posible leer la hoja: {exc}",
                })
                continue

            if h is None or h.empty:
                diagnostico.append({
                    "Archivo": nombre, "Hoja": hoja, "Estado": "VACÍA",
                    "Filas": 0, "Año asignado": ano_archivo,
                    "Detalle": "Hoja sin registros",
                })
                continue

            columnas = {
                destino: _buscar_columna(
                    h.columns,
                    opciones,
                    prefijos=["UNIDADES"] if destino == "UNIDADES" else None,
                )
                for destino, opciones in aliases.items()
            }
            obligatorias = ["CODIGO", "VENTA", "UNIDADES"]
            faltantes = [c for c in obligatorias if columnas[c] is None]
            mes_hoja = _inferir_mes(hoja)
            if columnas["MES"] is None and mes_hoja is None:
                faltantes.append("MES")

            if faltantes:
                diagnostico.append({
                    "Archivo": nombre, "Hoja": hoja, "Estado": "OMITIDA",
                    "Filas": len(h), "Año asignado": ano_archivo,
                    "Detalle": "Faltan columnas: " + ", ".join(faltantes),
                })
                continue

            temp = pd.DataFrame(index=h.index)
            temp["CODIGO"] = h[columnas["CODIGO"]]
            temp["VENTA"] = h[columnas["VENTA"]]
            temp["UNIDADES"] = h[columnas["UNIDADES"]]
            temp["AÑO"] = ano_archivo

            if columnas["MES"] is not None:
                mes_original = h[columnas["MES"]]
                mes_texto = mes_original.astype(str).map(_normalizar_encabezado)
                mes_numero = pd.to_numeric(mes_original, errors="coerce")
                temp["MES_NUM"] = mes_texto.map(MESES).fillna(mes_numero)
            else:
                temp["MES_NUM"] = mes_hoja

            for col in ["DESCRIPCION", "CATEGORIA", "FAMILIA", "TIPO"]:
                temp[col] = (
                    h[columnas[col]] if columnas[col] is not None else "SIN_DATO"
                )
            temp["COMBOS"] = (
                h[columnas["COMBOS"]] if columnas["COMBOS"] is not None else 0
            )

            temp["CODIGO"] = normalizar_codigo(temp["CODIGO"])
            temp["VENTA"] = pd.to_numeric(temp["VENTA"], errors="coerce")
            temp["UNIDADES"] = pd.to_numeric(temp["UNIDADES"], errors="coerce")
            temp["MES_NUM"] = pd.to_numeric(temp["MES_NUM"], errors="coerce")
            temp["COMBOS"] = pd.to_numeric(temp["COMBOS"], errors="coerce").fillna(0)

            temp = temp.dropna(subset=["CODIGO", "VENTA", "UNIDADES", "MES_NUM"])
            temp = temp[temp["MES_NUM"].between(1, 12)]
            temp["MES_NUM"] = temp["MES_NUM"].astype(int)
            temp["VENTA"] = temp["VENTA"].clip(lower=0)
            temp["UNIDADES"] = temp["UNIDADES"].clip(lower=0)
            temp["COMBOS"] = temp["COMBOS"].clip(0, 1).astype(int)
            temp["Periodo"] = temp["AÑO"] * 100 + temp["MES_NUM"]

            for col in ["DESCRIPCION", "CATEGORIA", "FAMILIA", "TIPO"]:
                temp[col] = temp[col].fillna("SIN_DATO").astype(str).str.strip()

            if temp.empty:
                diagnostico.append({
                    "Archivo": nombre, "Hoja": hoja, "Estado": "OMITIDA",
                    "Filas": 0, "Año asignado": ano_archivo,
                    "Detalle": "No quedaron filas válidas después de limpiar",
                })
                continue

            partes.append(temp)
            diagnostico.append({
                "Archivo": nombre, "Hoja": hoja, "Estado": "CARGADA",
                "Filas": len(temp), "Año asignado": ano_archivo,
                "Detalle": (
                    f"{temp['CODIGO'].nunique()} SKU | meses {sorted(temp['MES_NUM'].unique().tolist())} | "
                    f"unidades leídas desde: {columnas['UNIDADES']}"
                ),
            })

    if not partes:
        raise ValueError(
            "Ninguna hoja pudo cargarse. Abre el diagnóstico y revisa los encabezados."
        )

    df = pd.concat(partes, ignore_index=True)
    agrupado = (
        df.groupby(["CODIGO", "Periodo", "MES_NUM", "AÑO"], as_index=False)
        .agg(
            VENTA=("VENTA", "sum"),
            UNIDADES=("UNIDADES", "sum"),
            DESCRIPCION=("DESCRIPCION", "first"),
            CATEGORIA=("CATEGORIA", "first"),
            FAMILIA=("FAMILIA", "first"),
            TIPO=("TIPO", "first"),
            COMBOS=("COMBOS", "first"),
        )
    )
    agrupado["PRECIO_PROMEDIO"] = np.where(
        agrupado["UNIDADES"] > 0,
        agrupado["VENTA"] / agrupado["UNIDADES"],
        np.nan,
    )
    salida = completar_calendario(agrupado)
    return salida, pd.DataFrame(diagnostico)


def completar_calendario(df):
    salida = []
    for codigo, grupo in df.groupby("CODIGO", sort=False):
        x = grupo.sort_values("Periodo").copy()
        fechas = pd.to_datetime(
            dict(year=x["AÑO"], month=x["MES_NUM"], day=1)
        )
        x.index = fechas
        x = x.reindex(pd.date_range(fechas.min(), fechas.max(), freq="MS"))
        x["CODIGO"] = codigo
        x["AÑO"] = x.index.year
        x["MES_NUM"] = x.index.month
        x["Periodo"] = x["AÑO"] * 100 + x["MES_NUM"]
        x["VENTA"] = x["VENTA"].fillna(0.0)
        x["UNIDADES"] = x["UNIDADES"].fillna(0.0)
        for col in ["DESCRIPCION", "CATEGORIA", "FAMILIA", "TIPO", "COMBOS"]:
            x[col] = x[col].ffill().bfill()
        x["PRECIO_PROMEDIO"] = np.where(
            x["UNIDADES"] > 0, x["VENTA"] / x["UNIDADES"], np.nan
        )
        x["PRECIO_PROMEDIO"] = x["PRECIO_PROMEDIO"].ffill()
        salida.append(x.reset_index(drop=True))

    z = pd.concat(salida, ignore_index=True)
    mediana_categoria = z.groupby("CATEGORIA")["PRECIO_PROMEDIO"].transform("median")
    z["PRECIO_PROMEDIO"] = (
        z["PRECIO_PROMEDIO"]
        .fillna(mediana_categoria)
        .fillna(z["PRECIO_PROMEDIO"].median())
        .clip(lower=0)
    )
    z["COMBOS"] = pd.to_numeric(z["COMBOS"], errors="coerce").fillna(0).astype(int)
    return z


def crear_features(df, objetivo):
    z = df.sort_values(["CODIGO", "Periodo"]).copy()
    grupo = z.groupby("CODIGO")[objetivo]

    for lag in LAGS:
        z[f"lag{lag}"] = grupo.shift(lag)

    for ventana in [3, 6, 12]:
        z[f"mean{ventana}"] = grupo.transform(
            lambda s: s.shift(1).rolling(ventana, min_periods=2).mean()
        )
        z[f"std{ventana}"] = grupo.transform(
            lambda s: s.shift(1).rolling(ventana, min_periods=2).std()
        ).fillna(0)

    z["sin_mes"] = np.sin(2 * np.pi * z["MES_NUM"] / 12)
    z["cos_mes"] = np.cos(2 * np.pi * z["MES_NUM"] / 12)
    z["ANO"] = z["AÑO"]
    z["tendencia_3_12"] = np.where(
        z["mean12"].abs() > 1e-9, z["mean3"] / z["mean12"], 1.0
    )

    for col in CAT_FEATURES:
        z[col] = z[col].fillna("SIN_DATO").astype(str)
    return z


def nuevo_modelo():
    if CatBoostRegressor is None:
        raise ImportError("Instala CatBoost con: python -m pip install catboost")
    return CatBoostRegressor(
        iterations=500,
        depth=7,
        learning_rate=0.05,
        loss_function="MAE",
        eval_metric="MAE",
        l2_leaf_reg=10,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=4,
    )


def entrenar_objetivo(df, objetivo):
    f = crear_features(df, objetivo).dropna(subset=["lag1", objetivo]).copy()
    medianas = f[NUM_FEATURES].median().fillna(0)
    f[NUM_FEATURES] = f[NUM_FEATURES].fillna(medianas)
    pesos = np.where(f["MES_NUM"].isin([11, 12]), 2.5, 1.0)
    pesos *= 1 + 0.15 * (f["AÑO"] - f["AÑO"].min())

    modelo = nuevo_modelo()
    modelo.fit(
        f[FEATURES],
        np.log1p(f[objetivo].clip(lower=0)),
        cat_features=CAT_FEATURES,
        sample_weight=pesos,
    )
    return modelo, medianas


@st.cache_resource(show_spinner=False)
def entrenar_modelos(df):
    return {
        objetivo: entrenar_objetivo(df, objetivo)
        for objetivo in ["VENTA", "UNIDADES", "PRECIO_PROMEDIO"]
    }


def vector_futuro(historial, periodo, objetivo, medianas):
    hist = historial.sort_values("Periodo")
    valores = hist[objetivo].astype(float).tolist()
    fila = {
        f"lag{lag}": valores[-lag] if len(valores) >= lag else np.nan
        for lag in LAGS
    }

    for ventana in [3, 6, 12]:
        arreglo = np.asarray(valores[-ventana:], dtype=float)
        fila[f"mean{ventana}"] = arreglo.mean() if len(arreglo) >= 2 else np.nan
        fila[f"std{ventana}"] = arreglo.std(ddof=1) if len(arreglo) >= 2 else 0.0

    mes = periodo % 100
    ano = periodo // 100
    fila["sin_mes"] = np.sin(2 * np.pi * mes / 12)
    fila["cos_mes"] = np.cos(2 * np.pi * mes / 12)
    fila["MES_NUM"] = mes
    fila["ANO"] = ano
    fila["tendencia_3_12"] = (
        fila["mean3"] / fila["mean12"]
        if pd.notna(fila["mean12"]) and fila["mean12"] > 0
        else 1.0
    )

    ultimo = hist.iloc[-1]
    for col in CAT_FEATURES:
        fila[col] = str(ultimo.get(col, "SIN_DATO"))

    x = pd.DataFrame([fila], columns=FEATURES)
    x[NUM_FEATURES] = x[NUM_FEATURES].fillna(medianas)
    return x


def predecir_objetivo(modelo, medianas, historial, periodos, objetivo):
    hist = historial.copy()
    predicciones = []

    for periodo in periodos:
        pred = float(
            np.expm1(
                modelo.predict(vector_futuro(hist, periodo, objetivo, medianas))[0]
            )
        )
        pred = max(0.0, pred)
        predicciones.append(pred)

        nueva = hist.iloc[-1].copy()
        nueva["Periodo"] = periodo
        nueva["AÑO"] = periodo // 100
        nueva["MES_NUM"] = periodo % 100
        nueva[objetivo] = pred
        hist = pd.concat([hist, pd.DataFrame([nueva])], ignore_index=True)

    return predicciones


def indice_estacional(historial, ano_objetivo):
    h = historial[historial["AÑO"] < ano_objetivo]
    temporada = h[h["MES_NUM"].isin([11, 12])]["VENTA"]
    resto = h[~h["MES_NUM"].isin([11, 12])]["VENTA"]
    if len(temporada) < 2 or len(resto) < 6 or resto.mean() <= 0:
        return 1.0
    return float(temporada.mean() / resto.mean())


def crecimiento_ytd(historial, ano_objetivo):
    actual = historial[
        (historial["AÑO"] == ano_objetivo) & (historial["MES_NUM"] <= 10)
    ]
    previo = historial[
        (historial["AÑO"] == ano_objetivo - 1) & (historial["MES_NUM"] <= 10)
    ]
    meses = sorted(set(actual["MES_NUM"]) & set(previo["MES_NUM"]))
    if len(meses) < 6:
        return 1.0
    va = actual[actual["MES_NUM"].isin(meses)]["VENTA"].sum()
    vp = previo[previo["MES_NUM"].isin(meses)]["VENTA"].sum()
    return float(np.clip(va / vp, 0.80, 1.40)) if vp > 0 else 1.0


def aplicar_ajuste_estacional(pred_base, historial, periodo):
    """Ajuste estacional final validado: piso parcial 45% Nov / 60% Dic.

    No copia automáticamente el mismo mes anterior. Si la predicción queda por
    debajo del piso parcial, recupera solo una parte de la diferencia.
    """
    mes = periodo % 100
    ano = periodo // 100
    indice = indice_estacional(historial, ano)
    mismos = historial[
        (historial["MES_NUM"] == mes) & (historial["AÑO"] < ano)
    ].sort_values("AÑO").tail(2)

    diagnostico = {
        "Indice_estacional": float(indice),
        "Referencia_ultimo_ano": np.nan,
        "Crecimiento_YTD": 1.0,
        "Prediccion_estacional": float(pred_base),
        "Prediccion_combinada": float(pred_base),
        "Piso_parcial": np.nan,
        "Porcentaje_recuperacion": 0.0,
        "Correccion_piso": 0.0,
        "Ajuste_aplicado": False,
    }

    if mismos.empty or indice < UMBRAL_ESTACIONAL:
        return max(0.0, float(pred_base)), diagnostico

    valores = mismos["VENTA"].astype(float).to_numpy()
    ultimo_ano = float(valores[-1])
    referencia_historica = (
        ultimo_ano if len(valores) == 1
        else 0.65 * ultimo_ano + 0.35 * float(valores[-2])
    )
    crec = crecimiento_ytd(historial, ano)
    referencia_ytd = referencia_historica * crec

    prediccion_estacional = (
        PESO_FORECAST_ESTACIONAL * float(pred_base)
        + (1.0 - PESO_FORECAST_ESTACIONAL) * referencia_ytd
    )

    crecimiento_suavizado = float(np.clip(
        1.0 + 0.50 * (crec - 1.0), 0.90, 1.20
    ))
    referencia_suavizada = ultimo_ano * crecimiento_suavizado
    prediccion_combinada = (
        0.90 * prediccion_estacional + 0.10 * referencia_suavizada
    )

    factor_piso = float(np.clip(
        1.0 + 0.35 * (crec - 1.0), 0.94, 1.15
    ))
    piso_parcial = ultimo_ano * factor_piso

    porcentaje_recuperacion = 0.45 if mes == 11 else 0.60 if mes == 12 else 0.40
    if prediccion_combinada < piso_parcial:
        correccion_piso = porcentaje_recuperacion * (
            piso_parcial - prediccion_combinada
        )
        prediccion_final = prediccion_combinada + correccion_piso
    else:
        correccion_piso = 0.0
        prediccion_final = prediccion_combinada

    diagnostico.update({
        "Referencia_ultimo_ano": ultimo_ano,
        "Crecimiento_YTD": float(crec),
        "Prediccion_estacional": float(prediccion_estacional),
        "Prediccion_combinada": float(prediccion_combinada),
        "Piso_parcial": float(piso_parcial),
        "Porcentaje_recuperacion": float(porcentaje_recuperacion),
        "Correccion_piso": float(correccion_piso),
        "Ajuste_aplicado": True,
    })
    return max(0.0, float(prediccion_final)), diagnostico

@st.cache_data(show_spinner=False)
def generar_forecast(df, version_modelo="final-45-60-v1"):
    modelos = entrenar_modelos(df)
    filas = []
    periodos = [202611, 202612]

    for codigo, historial in df.groupby("CODIGO", sort=False):
        hist = historial.sort_values("Periodo")
        if hist.empty:
            continue
        # Solo pronosticar SKU activos con información reciente de 2026.
        # Evita saltar directamente de 2024/2025 a noviembre de 2026.
        ultimo_periodo = int(hist["Periodo"].max())
        if ultimo_periodo < 202608:
            continue
        meta = hist.iloc[-1]

        venta_directa = predecir_objetivo(
            *modelos["VENTA"], hist, periodos, "VENTA"
        )
        unidades = predecir_objetivo(
            *modelos["UNIDADES"], hist, periodos, "UNIDADES"
        )
        precio = predecir_objetivo(
            *modelos["PRECIO_PROMEDIO"], hist, periodos, "PRECIO_PROMEDIO"
        )

        for periodo, vd, unid, precio_prom in zip(
            periodos, venta_directa, unidades, precio
        ):
            estructural = unid * precio_prom
            base = (
                PESO_VENTA_DIRECTA * vd
                + PESO_ESTRUCTURAL * estructural
            )
            final, diag = aplicar_ajuste_estacional(base, hist, periodo)
            filas.append({
                "CODIGO": codigo,
                "DESCRIPCION": str(meta.get("DESCRIPCION", "")),
                "CATEGORIA": str(meta.get("CATEGORIA", "SIN_DATO")).strip(),
                "COMBOS": int(meta.get("COMBOS", 0)),
                "Periodo": periodo,
                "Venta_directa": float(vd),
                "Unidades_pred": float(unid),
                "Precio_pred": float(precio_prom),
                "Venta_estructural": float(estructural),
                "Prediccion_base": float(base),
                "Referencia_ultimo_ano": diag["Referencia_ultimo_ano"],
                "Crecimiento_YTD": diag["Crecimiento_YTD"],
                "Prediccion_estacional": diag["Prediccion_estacional"],
                "Prediccion_combinada": diag["Prediccion_combinada"],
                "Piso_parcial": diag["Piso_parcial"],
                "Porcentaje_recuperacion": diag["Porcentaje_recuperacion"],
                "Correccion_piso": diag["Correccion_piso"],
                "Indice_estacional": diag["Indice_estacional"],
                "Ajuste_aplicado": diag["Ajuste_aplicado"],
                "Prediccion_final": float(final),
            })

    return pd.DataFrame(filas)


def preparar_grafica(df_historico, forecast, categoria, combo, producto):
    """Prepara histórico comparable y una línea forecast conectada a octubre 2026."""
    futuro = forecast.copy()

    if categoria != "TODAS":
        futuro = futuro[futuro["CATEGORIA"] == categoria]
    if combo != "TODOS":
        valor_combo = 1 if combo == "COMBOS" else 0
        futuro = futuro[futuro["COMBOS"] == valor_combo]
    if producto != "TODOS":
        futuro = futuro[futuro["CODIGO"].astype(str) == producto]

    # El histórico se restringe al mismo universo de SKU que sí tienen forecast.
    codigos_activos = futuro["CODIGO"].astype(str).unique().tolist()
    hist = df_historico[df_historico["CODIGO"].astype(str).isin(codigos_activos)].copy()
    if hist.empty or futuro.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    historico = hist.groupby(["AÑO", "MES_NUM"], as_index=False)["VENTA"].sum()
    historico["Fecha"] = pd.to_datetime(dict(
        year=historico["AÑO"].astype(int),
        month=historico["MES_NUM"].astype(int), day=1
    ))
    historico = historico[historico["Fecha"] <= pd.Timestamp("2026-10-01")]
    historico = historico.rename(columns={"VENTA": "Valor"})
    historico["Serie"] = "Venta real"

    previsto = futuro.groupby("Periodo", as_index=False)["Prediccion_final"].sum()
    previsto["Fecha"] = pd.to_datetime(dict(
        year=(previsto["Periodo"] // 100).astype(int),
        month=(previsto["Periodo"] % 100).astype(int), day=1
    ))
    previsto = previsto.rename(columns={"Prediccion_final": "Valor"})
    previsto["Serie"] = "Forecast"

    # Punto puente: octubre real se agrega al forecast para conectar las líneas.
    octubre = historico[historico["Fecha"] == pd.Timestamp("2026-10-01")][["Fecha", "Valor"]].copy()
    if octubre.empty:
        ultima = historico.sort_values("Fecha").tail(1)[["Fecha", "Valor"]].copy()
        puente = ultima
    else:
        puente = octubre
    puente["Serie"] = "Forecast"
    forecast_conectado = pd.concat(
        [puente[["Fecha", "Valor", "Serie"]], previsto[["Fecha", "Valor", "Serie"]]],
        ignore_index=True,
    ).sort_values("Fecha")

    return (
        historico[["Fecha", "Valor", "Serie"]].sort_values("Fecha"),
        forecast_conectado,
        previsto[["Fecha", "Valor", "Serie"]].sort_values("Fecha"),
    )


def diagnosticar_universo_sku(df):
    """Explica la diferencia entre universo 2026 y backtest Nov-Dic 2025.

    El universo parte de todos los códigos presentes en 2026. La elegibilidad
    replica las condiciones del backtest: historial hasta octubre 2025,
    observaciones Nov-Dic 2025 y exclusión de combos.
    """
    filas = []
    universo_2026 = sorted(
        df.loc[df["AÑO"] == 2026, "CODIGO"].astype(str).unique().tolist()
    )

    for codigo in universo_2026:
        producto = df[df["CODIGO"].astype(str) == codigo].sort_values("Periodo")
        ultimo = producto.iloc[-1]
        periodos = set(producto["Periodo"].astype(int).tolist())
        historia_previa = producto[producto["Periodo"] < 202511]
        nov_dic_2025 = producto[producto["Periodo"].isin([202511, 202512])]
        es_combo = bool(pd.to_numeric(producto["COMBOS"], errors="coerce").fillna(0).astype(int).eq(1).any())
        tiene_historia = not historia_previa.empty
        tiene_noviembre = 202511 in periodos
        tiene_diciembre = 202512 in periodos
        tiene_ambos = tiene_noviembre and tiene_diciembre
        venta_real_nov_dic = float(nov_dic_2025["VENTA"].sum()) if not nov_dic_2025.empty else 0.0
        primer_periodo = int(producto["Periodo"].min())
        ultimo_periodo = int(producto["Periodo"].max())
        activo_forecast_2026 = ultimo_periodo >= 202608

        elegible_con_combo = tiene_historia and tiene_ambos
        elegible_sin_combo = elegible_con_combo and not es_combo

        motivos = []
        if not tiene_historia:
            motivos.append("Sin historial antes de noviembre 2025")
        if not tiene_ambos:
            motivos.append("Sin ambos meses reales Nov-Dic 2025")
        if es_combo:
            motivos.append("Combo excluido del backtest")
        if venta_real_nov_dic <= 0:
            motivos.append("Venta Nov-Dic 2025 igual a cero")
        if not activo_forecast_2026:
            motivos.append("Sin actividad reciente para forecast 2026")
        if not motivos:
            motivos.append("Incluido en backtest sin combos")

        if elegible_sin_combo:
            motivo_principal = "Incluido en backtest sin combos"
        elif es_combo and elegible_con_combo:
            motivo_principal = "Combo excluido del backtest"
        elif not tiene_historia:
            motivo_principal = "Producto nuevo o sin historial previo"
        elif not tiene_ambos:
            motivo_principal = "Sin ambos meses Nov-Dic 2025"
        else:
            motivo_principal = "No elegible por calidad/disponibilidad"

        filas.append({
            "CODIGO": codigo,
            "DESCRIPCION": str(ultimo.get("DESCRIPCION", "")),
            "CATEGORIA": str(ultimo.get("CATEGORIA", "SIN_DATO")),
            "COMBOS": int(es_combo),
            "Primer_periodo": primer_periodo,
            "Ultimo_periodo": ultimo_periodo,
            "Tiene_historia_pre_Nov2025": tiene_historia,
            "Tiene_Nov2025": tiene_noviembre,
            "Tiene_Dic2025": tiene_diciembre,
            "Venta_real_NovDic2025": venta_real_nov_dic,
            "Elegible_con_combos": elegible_con_combo,
            "Elegible_sin_combos": elegible_sin_combo,
            "Activo_para_forecast_2026": activo_forecast_2026,
            "Motivo_principal": motivo_principal,
            "Todos_los_motivos": " | ".join(motivos),
        })

    detalle = pd.DataFrame(filas)
    if detalle.empty:
        return detalle, pd.DataFrame()

    resumen = pd.DataFrame([
        {"Condición": "Universo de SKU en 2026", "Cantidad_SKU": detalle["CODIGO"].nunique()},
        {"Condición": "Con historial antes de noviembre 2025", "Cantidad_SKU": int(detalle["Tiene_historia_pre_Nov2025"].sum())},
        {"Condición": "Con noviembre y diciembre 2025", "Cantidad_SKU": int((detalle["Tiene_Nov2025"] & detalle["Tiene_Dic2025"]).sum())},
        {"Condición": "Evaluables incluyendo combos", "Cantidad_SKU": int(detalle["Elegible_con_combos"].sum())},
        {"Condición": "Evaluables sin combos", "Cantidad_SKU": int(detalle["Elegible_sin_combos"].sum())},
        {"Condición": "No evaluables vs universo 2026", "Cantidad_SKU": int((~detalle["Elegible_sin_combos"]).sum())},
        {"Condición": "Activos para forecast Nov-Dic 2026", "Cantidad_SKU": int(detalle["Activo_para_forecast_2026"].sum())},
    ])
    return detalle, resumen


def resumen_motivos_exclusion(detalle):
    excluidos = detalle[~detalle["Elegible_sin_combos"]].copy()
    if excluidos.empty:
        return pd.DataFrame()
    return (
        excluidos.groupby("Motivo_principal", as_index=False)
        .agg(Cantidad_SKU=("CODIGO", "nunique"))
        .sort_values("Cantidad_SKU", ascending=False)
    )


archivos = st.file_uploader(
    "Carga los archivos de ventas 2024, 2025 y 2026",
    type=["xlsx"],
    accept_multiple_files=True,
)

if archivos:
    try:
        # La carga no usa caché: cada ejecución procesa exactamente los archivos visibles.
        df, diagnostico_carga = cargar_archivos(archivos)
        resumen_anual = (
            df.groupby("AÑO", as_index=False)
            .agg(SKU=("CODIGO", "nunique"), Registros=("Periodo", "size"),
                 Desde=("Periodo", "min"), Hasta=("Periodo", "max"))
        )
        anos_cargados = set(resumen_anual["AÑO"].astype(int).tolist())
        st.success("Los archivos fueron leídos nuevamente sin utilizar caché.")
        st.dataframe(resumen_anual, use_container_width=True, hide_index=True)
        with st.expander("Diagnóstico detallado de cada hoja"):
            st.dataframe(diagnostico_carga, use_container_width=True, hide_index=True)
        faltantes = sorted({2024, 2025, 2026} - anos_cargados)
        if faltantes:
            st.error(
                f"No se cargaron datos válidos de {faltantes}. Revisa en el diagnóstico "
                "qué columna falta en las hojas correspondientes."
            )
            st.stop()

        with st.expander("Diagnóstico de los SKU no incluidos en el backtest", expanded=False):
            detalle_sku, resumen_sku = diagnosticar_universo_sku(df)
            st.markdown(
                "El universo 2026 y el backtest 2025 son poblaciones distintas. "
                "Esta sección reproduce las reglas de elegibilidad y explica cada exclusión."
            )
            st.dataframe(resumen_sku, use_container_width=True, hide_index=True)

            motivos_sku = resumen_motivos_exclusion(detalle_sku)
            st.markdown("#### Motivos principales de exclusión")
            st.dataframe(motivos_sku, use_container_width=True, hide_index=True)

            solo_faltantes = st.checkbox(
                "Mostrar únicamente SKU no evaluables",
                value=True,
                key="solo_sku_faltantes",
            )
            filtro_motivo = st.selectbox(
                "Filtrar motivo",
                ["TODOS"] + motivos_sku["Motivo_principal"].astype(str).tolist(),
                key="filtro_motivo_sku",
            )
            tabla_sku = detalle_sku.copy()
            if solo_faltantes:
                tabla_sku = tabla_sku[~tabla_sku["Elegible_sin_combos"]]
            if filtro_motivo != "TODOS":
                tabla_sku = tabla_sku[tabla_sku["Motivo_principal"] == filtro_motivo]

            st.caption(
                f"Se muestran {tabla_sku['CODIGO'].nunique():,} SKU. "
                "Un SKU puede tener más de un motivo en la columna Todos_los_motivos."
            )
            st.dataframe(
                tabla_sku.style.format({"Venta_real_NovDic2025": "${:,.2f}"}),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Descargar diagnóstico completo de SKU",
                detalle_sku.to_csv(index=False).encode("utf-8-sig"),
                "diagnostico_universo_306_vs_backtest_221.csv",
                "text/csv",
            )
            st.download_button(
                "Descargar solo SKU no evaluables",
                detalle_sku[~detalle_sku["Elegible_sin_combos"]]
                .to_csv(index=False).encode("utf-8-sig"),
                "sku_no_incluidos_backtest_nov_dic_2025.csv",
                "text/csv",
            )

        col_boton, col_estado = st.columns([1, 2])
        with col_boton:
            if st.button("Limpiar forecast anterior"):
                st.session_state.pop("forecast_2026_final_45_60", None)
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
        with col_estado:
            st.info("Modelo final activo: recuperación 45% en noviembre y 60% en diciembre.")

        if "forecast_2026_final_45_60" not in st.session_state:
            st.info("Genera el forecast para habilitar la gráfica de noviembre-diciembre de 2026.")

        if st.button("Generar forecast final Nov-Dic 2026", type="primary"):
            with st.spinner("Entrenando el modelo y generando el forecast..."):
                st.session_state["forecast_2026_final_45_60"] = generar_forecast(df)
            st.rerun()

        if "forecast_2026_final_45_60" in st.session_state:
            forecast = st.session_state["forecast_2026_final_45_60"]
            if forecast.empty:
                st.error("No se encontraron SKU activos con datos recientes de 2026.")
                st.stop()

            st.markdown("### Filtros")
            categorias = ["TODAS"] + sorted(
                forecast["CATEGORIA"].dropna().astype(str).str.strip().unique().tolist()
            )
            c1, c2, c3 = st.columns([2, 1.25, 3])
            with c1:
                categoria = st.selectbox("Categoría", categorias)
            with c2:
                combo = st.segmented_control(
                    "Tipo de producto",
                    options=["TODOS", "SIN COMBOS", "COMBOS"],
                    default="SIN COMBOS",
                )
                if combo is None:
                    combo = "SIN COMBOS"

            universo = forecast.copy()
            if categoria != "TODAS":
                universo = universo[universo["CATEGORIA"] == categoria]
            if combo != "TODOS":
                universo = universo[
                    universo["COMBOS"] == (1 if combo == "COMBOS" else 0)
                ]

            productos = universo[["CODIGO", "DESCRIPCION"]].drop_duplicates()
            productos["ETIQUETA"] = (
                productos["CODIGO"].astype(str) + " | "
                + productos["DESCRIPCION"].astype(str)
            )
            mapa_productos = dict(
                zip(productos["ETIQUETA"], productos["CODIGO"].astype(str))
            )
            with c3:
                seleccion_producto = st.selectbox(
                    "Producto", ["TODOS"] + sorted(mapa_productos)
                )
            producto = "TODOS" if seleccion_producto == "TODOS" else mapa_productos[seleccion_producto]

            historico, forecast_linea, forecast_puntos = preparar_grafica(
                df, forecast, categoria, combo, producto
            )
            if historico.empty or forecast_puntos.empty:
                st.warning("No hay datos comparables para la selección actual.")
                st.stop()

            nov = forecast_puntos.loc[
                forecast_puntos["Fecha"] == pd.Timestamp("2026-11-01"), "Valor"
            ].sum()
            dic = forecast_puntos.loc[
                forecast_puntos["Fecha"] == pd.Timestamp("2026-12-01"), "Valor"
            ].sum()
            oct_real = historico.loc[
                historico["Fecha"] == pd.Timestamp("2026-10-01"), "Valor"
            ].sum()
            crecimiento_nov = (nov / oct_real - 1) * 100 if oct_real > 0 else np.nan
            crecimiento_dic = (dic / nov - 1) * 100 if nov > 0 else np.nan

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Octubre 2026 real", f"${oct_real:,.0f}")
            k2.metric("Noviembre 2026", f"${nov:,.0f}",
                      f"{crecimiento_nov:+.1f}% vs octubre" if pd.notna(crecimiento_nov) else None)
            k3.metric("Diciembre 2026", f"${dic:,.0f}",
                      f"{crecimiento_dic:+.1f}% vs noviembre" if pd.notna(crecimiento_dic) else None)
            k4.metric("SKU incluidos", f"{universo['CODIGO'].nunique():,}")

            fecha_min = pd.Timestamp("2024-01-01")
            fecha_max = pd.Timestamp("2026-12-01")
            eje_x = alt.X(
                "Fecha:T", title=None,
                scale=alt.Scale(domain=[fecha_min, fecha_max]),
                axis=alt.Axis(
                    format="%b %Y", labelAngle=-35,
                    tickCount={"interval": "month", "step": 2},
                    grid=False, labelPadding=10,
                ),
            )
            eje_y = alt.Y(
                "Valor:Q", title="Venta mensual",
                scale=alt.Scale(zero=False, nice=True),
                axis=alt.Axis(format="$,.2s", grid=True, gridOpacity=0.14),
            )
            tooltip = [
                alt.Tooltip("Fecha:T", title="Mes", format="%B %Y"),
                alt.Tooltip("Valor:Q", title="Venta", format="$,.2f"),
            ]

            linea_real = alt.Chart(historico).mark_line(
                color="#3B82F6", strokeWidth=3.2, interpolate="monotone"
            ).encode(x=eje_x, y=eje_y, tooltip=tooltip)
            puntos_reales = alt.Chart(historico).mark_circle(
                color="#3B82F6", size=44, opacity=0.9
            ).encode(x=eje_x, y=eje_y, tooltip=tooltip)

            linea_forecast = alt.Chart(forecast_linea).mark_line(
                color="#F59E0B", strokeWidth=3.5,
                strokeDash=[8, 5], interpolate="monotone"
            ).encode(x=eje_x, y=eje_y, tooltip=tooltip)
            puntos_forecast = alt.Chart(forecast_puntos).mark_circle(
                color="#F59E0B", size=125, stroke="white", strokeWidth=1.5
            ).encode(x=eje_x, y=eje_y, tooltip=tooltip)

            limite = pd.DataFrame({"Fecha": [pd.Timestamp("2026-11-01")]})
            regla = alt.Chart(limite).mark_rule(
                color="#94A3B8", strokeDash=[4, 4], opacity=0.75
            ).encode(x="Fecha:T")
            etiqueta = alt.Chart(pd.DataFrame({
                "Fecha": [pd.Timestamp("2026-11-01")],
                "Texto": ["Inicio forecast"]
            })).mark_text(
                align="left", dx=7, dy=-205,
                color="#94A3B8", fontSize=12
            ).encode(x="Fecha:T", text="Texto:N")

            chart = alt.layer(
                linea_real, puntos_reales,
                linea_forecast, puntos_forecast,
                regla, etiqueta,
            ).properties(
                height=520,
                title=alt.TitleParams(
                    text="Evolución mensual y forecast Nov-Dic 2026",
                    subtitle="Línea azul: ventas reales | Línea naranja punteada: pronóstico",
                    anchor="start", fontSize=18, subtitleFontSize=12,
                    offset=16,
                ),
            ).configure_view(strokeWidth=0).configure_axis(
                labelFontSize=12, titleFontSize=13
            )

            st.altair_chart(chart, use_container_width=True)
            st.caption(
                "El histórico utiliza exactamente los mismos SKU activos incluidos en el forecast. "
                "Los productos sin actividad reciente en 2026 se excluyen para evitar saltos artificiales."
            )
            st.download_button(
                "Descargar forecast filtrado CSV",
                universo.to_csv(index=False).encode("utf-8-sig"),
                "forecast_nov_dic_2026_45_60.csv",
                "text/csv",
            )
            with st.expander("Diagnóstico del ajuste 45%/60%"):
                st.caption(
                    "Predicción base muestra la salida 85% venta directa + 15% unidades por precio. "
                    "Corrección piso indica cuánto añadió la protección estacional; cero significa que el modelo ya estaba por encima del piso."
                )
                diag = universo[[
                    "CODIGO", "DESCRIPCION", "CATEGORIA", "COMBOS", "Periodo",
                    "Prediccion_base", "Referencia_ultimo_ano", "Crecimiento_YTD",
                    "Prediccion_estacional", "Prediccion_combinada", "Piso_parcial",
                    "Porcentaje_recuperacion", "Correccion_piso",
                    "Ajuste_aplicado", "Prediccion_final",
                ]].copy()
                st.dataframe(diag, use_container_width=True, hide_index=True)


    except Exception as error:
        st.exception(error)
else:
    st.info("Carga los tres archivos Excel para comenzar.")


