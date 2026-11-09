import io
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

st.set_page_config(page_title="Forecast venta, unidades y precio", layout="wide")
st.title("Forecast CatBoost: venta + unidades x precio etx")

MESES = {m: i + 1 for i, m in enumerate([
    "ENERO", "FEBRERO", "MARZO", "ABRIL", "MAYO", "JUNIO",
    "JULIO", "AGOSTO", "SEPTIEMBRE", "OCTUBRE", "NOVIEMBRE", "DICIEMBRE"
])}
CAT_FEATURES = ["CODIGO", "CATEGORIA", "FAMILIA", "TIPO", "COMBOS"]
LAGS = [1, 2, 3, 6, 12, 24]
BASE_NUM = ([f"lag{x}" for x in LAGS] +
            ["mean3", "std3", "mean6", "std6", "mean12", "std12",
             "sin_mes", "cos_mes", "MES_NUM", "ANO", "tendencia_3_12"])
FEATURES = BASE_NUM + CAT_FEATURES


def _texto_codigo(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


@st.cache_data(show_spinner=False)
def cargar_archivos(contenidos, nombres):
    partes = []
    diagnostico = []
    base_requerida = {"CODIGO", "VENTA", "MES", "AÑO"}
    utiles = base_requerida | {"UNIDADES", "PRECIO PROMEDIO", "DESCRIPCION",
                              "CATEGORIA", "FAMILIA", "TIPO", "COMBOS"}
    for contenido, nombre in zip(contenidos, nombres):
        hojas = pd.read_excel(io.BytesIO(contenido), sheet_name=None, engine="openpyxl")
        hojas_aceptadas = 0
        filas_archivo = 0
        anos_archivo = set()
        for hoja, h in hojas.items():
            if h is None or h.empty:
                continue
            h = h.copy()
            h.columns = [str(c).strip().upper() for c in h.columns]

            # Acepta el nombre nuevo y, por seguridad, corrige archivos antiguos.
            candidatos_unidades = [c for c in h.columns if c == "UNIDADES" or c.startswith("UNIDADES ")]
            if "UNIDADES" not in h.columns and candidatos_unidades:
                h = h.rename(columns={candidatos_unidades[0]: "UNIDADES"})

            faltantes = sorted(base_requerida - set(h.columns))
            if faltantes or "UNIDADES" not in h.columns:
                diagnostico.append({"Archivo": nombre, "Hoja": hoja, "Estado": "Omitida",
                    "Detalle": "Faltan: " + ", ".join(faltantes + ([] if "UNIDADES" in h.columns else ["UNIDADES"]))})
                continue

            temp = h[[c for c in h.columns if c in utiles]].copy()
            temp["ARCHIVO_ORIGEN"] = nombre
            temp["HOJA_ORIGEN"] = hoja
            partes.append(temp)
            hojas_aceptadas += 1
            filas_archivo += len(temp)
            anos_archivo.update(pd.to_numeric(temp["AÑO"], errors="coerce").dropna().astype(int).unique())

        diagnostico.append({"Archivo": nombre, "Hoja": "TOTAL", "Estado": "Cargado" if hojas_aceptadas else "Sin datos",
                            "Detalle": f"{hojas_aceptadas} hojas, {filas_archivo:,} filas, años {sorted(anos_archivo)}"})

    if not partes:
        raise ValueError("No se encontraron hojas válidas con CODIGO, VENTA, UNIDADES, MES y AÑO.")

    df = pd.concat(partes, ignore_index=True)
    df["CODIGO"] = _texto_codigo(df["CODIGO"])
    df["MES_NUM"] = df["MES"].astype(str).str.strip().str.upper().map(MESES)
    df["AÑO"] = pd.to_numeric(df["AÑO"], errors="coerce")
    for c in ["VENTA", "UNIDADES", "PRECIO PROMEDIO"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["VENTA"] = df["VENTA"].fillna(0).clip(lower=0)
    df["UNIDADES"] = df["UNIDADES"].fillna(0).clip(lower=0)
    if "COMBOS" not in df.columns:
        df["COMBOS"] = 0
    df["COMBOS"] = pd.to_numeric(df["COMBOS"], errors="coerce").fillna(0).clip(0, 1).astype(int)
    df = df.dropna(subset=["CODIGO", "MES_NUM", "AÑO"])
    df["AÑO"] = df["AÑO"].astype(int)
    df["Periodo"] = df["AÑO"] * 100 + df["MES_NUM"].astype(int)

    meta_cols = [c for c in ["DESCRIPCION", "CATEGORIA", "FAMILIA", "TIPO", "COMBOS"] if c in df]
    agg = {"VENTA": "sum", "UNIDADES": "sum", **{c: "first" for c in meta_cols}}
    g = df.groupby(["CODIGO", "Periodo", "MES_NUM", "AÑO"], as_index=False).agg(agg)
    g["PRECIO_PROMEDIO"] = np.where(g["UNIDADES"] > 0, g["VENTA"] / g["UNIDADES"], np.nan)
    for c in ["DESCRIPCION", "CATEGORIA", "FAMILIA", "TIPO"]:
        if c not in g:
            g[c] = "SIN_DATO"
        g[c] = g[c].fillna("SIN_DATO").astype(str)
    return completar_calendario(g), pd.DataFrame(diagnostico)


def completar_calendario(df):
    salida = []
    for codigo, x in df.groupby("CODIGO"):
        x = x.sort_values("Periodo").copy()
        fechas = pd.to_datetime(x["AÑO"].astype(str) + "-" + x["MES_NUM"].astype(str) + "-01")
        x.index = fechas
        x = x.reindex(pd.date_range(fechas.min(), fechas.max(), freq="MS"))
        x["CODIGO"] = codigo
        x["AÑO"] = x.index.year
        x["MES_NUM"] = x.index.month
        x["Periodo"] = x["AÑO"] * 100 + x["MES_NUM"]
        x["VENTA"] = x["VENTA"].fillna(0.0)
        x["UNIDADES"] = x["UNIDADES"].fillna(0.0)
        for c in ["DESCRIPCION", "CATEGORIA", "FAMILIA", "TIPO", "COMBOS"]:
            x[c] = x[c].ffill().bfill()
        x["PRECIO_PROMEDIO"] = np.where(x["UNIDADES"] > 0, x["VENTA"] / x["UNIDADES"], np.nan)
        # Precio faltante se imputa solo desde el pasado del SKU, luego mediana de categoria.
        x["PRECIO_PROMEDIO"] = x["PRECIO_PROMEDIO"].ffill()
        salida.append(x.reset_index(drop=True))
    z = pd.concat(salida, ignore_index=True)
    med_cat = z.groupby("CATEGORIA")["PRECIO_PROMEDIO"].transform("median")
    z["PRECIO_PROMEDIO"] = z["PRECIO_PROMEDIO"].fillna(med_cat).fillna(z["PRECIO_PROMEDIO"].median()).clip(lower=0)
    return z


def crear_features_objetivo(df, objetivo):
    z = df.sort_values(["CODIGO", "Periodo"]).copy()
    grp = z.groupby("CODIGO")[objetivo]
    for lag in LAGS:
        z[f"lag{lag}"] = grp.shift(lag)
    for w in [3, 6, 12]:
        z[f"mean{w}"] = grp.transform(lambda s: s.shift(1).rolling(w, min_periods=2).mean())
        z[f"std{w}"] = grp.transform(lambda s: s.shift(1).rolling(w, min_periods=2).std()).fillna(0)
    z["sin_mes"] = np.sin(2 * np.pi * z["MES_NUM"] / 12)
    z["cos_mes"] = np.cos(2 * np.pi * z["MES_NUM"] / 12)
    z["ANO"] = z["AÑO"]
    z["tendencia_3_12"] = np.where(z["mean12"].abs() > 1e-9, z["mean3"] / z["mean12"], 1.0)
    for c in CAT_FEATURES:
        z[c] = z[c].fillna("SIN_DATO").astype(str)
    return z


def nuevo_modelo(iterations=650):
    if CatBoostRegressor is None:
        raise ImportError("Instala CatBoost: python -m pip install catboost")
    return CatBoostRegressor(iterations=iterations, depth=8, learning_rate=0.04,
        loss_function="MAE", eval_metric="MAE", l2_leaf_reg=10,
        random_seed=42, verbose=False, allow_writing_files=False, thread_count=-1)


def entrenar_objetivo(df, objetivo):
    f = crear_features_objetivo(df, objetivo).dropna(subset=["lag1", objetivo]).copy()
    medianas = f[BASE_NUM].median().fillna(0)
    f[BASE_NUM] = f[BASE_NUM].fillna(medianas)
    y = np.log1p(f[objetivo].clip(lower=0))
    pesos = np.where(f["MES_NUM"].isin([11, 12]), 2.5, 1.0)
    pesos *= 1 + 0.15 * (f["AÑO"] - f["AÑO"].min())
    m = nuevo_modelo()
    m.fit(f[FEATURES], y, cat_features=CAT_FEATURES, sample_weight=pesos)
    return m, medianas


def vector_futuro(hist, periodo, objetivo, medianas):
    hist = hist.sort_values("Periodo")
    vals = hist[objetivo].astype(float).tolist()
    q = {f"lag{lag}": vals[-lag] if len(vals) >= lag else np.nan for lag in LAGS}
    for w in [3, 6, 12]:
        a = np.asarray(vals[-w:], float)
        q[f"mean{w}"] = a.mean() if len(a) >= 2 else np.nan
        q[f"std{w}"] = a.std(ddof=1) if len(a) >= 2 else 0.0
    mes, ano = periodo % 100, periodo // 100
    q.update({"sin_mes": np.sin(2*np.pi*mes/12), "cos_mes": np.cos(2*np.pi*mes/12),
              "MES_NUM": mes, "ANO": ano})
    q["tendencia_3_12"] = q["mean3"] / q["mean12"] if pd.notna(q["mean12"]) and q["mean12"] > 0 else 1.0
    ultimo = hist.iloc[-1]
    for c in CAT_FEATURES:
        q[c] = str(ultimo.get(c, "SIN_DATO"))
    X = pd.DataFrame([q], columns=FEATURES)
    X[BASE_NUM] = X[BASE_NUM].fillna(medianas)
    return X


def predecir_objetivo(modelo, medianas, hist, periodos, objetivo):
    h = hist.copy()
    preds = []
    for periodo in periodos:
        pred = max(0.0, float(np.expm1(modelo.predict(vector_futuro(h, periodo, objetivo, medianas))[0])))
        preds.append(pred)
        fila = h.iloc[-1].copy()
        fila["Periodo"], fila["AÑO"], fila["MES_NUM"], fila[objetivo] = periodo, periodo//100, periodo%100, pred
        h = pd.concat([h, pd.DataFrame([fila])], ignore_index=True)
    return preds


def entrenar_tres_modelos(df):
    return {obj: entrenar_objetivo(df, obj) for obj in ["VENTA", "UNIDADES", "PRECIO_PROMEDIO"]}


def pronosticar_sku(modelos, hist, periodos, peso_directo=0.65):
    pv = predecir_objetivo(*modelos["VENTA"], hist, periodos, "VENTA")
    pu = predecir_objetivo(*modelos["UNIDADES"], hist, periodos, "UNIDADES")
    pp = predecir_objetivo(*modelos["PRECIO_PROMEDIO"], hist, periodos, "PRECIO_PROMEDIO")
    estructural = [u * p for u, p in zip(pu, pp)]
    final = [peso_directo*v + (1-peso_directo)*e for v, e in zip(pv, estructural)]
    return pv, pu, pp, estructural, final


def metricas(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    den = max(np.abs(y).sum(), 1.0)
    return {"WAPE": 100*np.abs(y-p).sum()/den,
            "MAE": mean_absolute_error(y,p),
            "RMSE": mean_squared_error(y,p)**0.5,
            "R2": r2_score(y,p) if len(y)>1 and np.var(y)>0 else np.nan,
            "SESGO": 100*(p.sum()-y.sum())/max(abs(y.sum()),1.0)}


def backtest(df, periodos, peso_directo):
    corte = min(periodos)
    train, real = df[df["Periodo"] < corte].copy(), df[df["Periodo"].isin(periodos)].copy()
    modelos = entrenar_tres_modelos(train)
    filas = []
    for codigo in sorted(set(train.CODIGO) & set(real.CODIGO)):
        h = train[train.CODIGO == codigo]
        r = real[real.CODIGO == codigo].sort_values("Periodo")
        ps = r.Periodo.astype(int).tolist()
        if not ps: continue
        pv, pu, pp, pe, pf = pronosticar_sku(modelos, h, ps, peso_directo)
        for (_, rr), a,b,c,d,e in zip(r.iterrows(), pv,pu,pp,pe,pf):
            filas.append({"CODIGO": codigo, "Periodo": int(rr.Periodo), "Actual": rr.VENTA,
                          "Venta_directa": a, "Unidades_pred": b, "Precio_pred": c,
                          "Venta_estructural": d, "Prediccion": e, "CATEGORIA": rr.CATEGORIA})
    return pd.DataFrame(filas)


# =========================================================
# AJUSTE ESTACIONAL + CALIBRACION ANTISESGO REGULARIZADA
# =========================================================
def indice_estacional(hist, anio_objetivo):
    h = hist[hist["AÑO"] < anio_objetivo]
    temp = h[h["MES_NUM"].isin([11, 12])]["VENTA"]
    resto = h[~h["MES_NUM"].isin([11, 12])]["VENTA"]
    if len(temp) < 2 or len(resto) < 6 or resto.mean() <= 0:
        return 1.0
    return float(temp.mean() / resto.mean())


def crecimiento_ytd(hist, anio_objetivo):
    a = hist[(hist["AÑO"] == anio_objetivo) & (hist["MES_NUM"] <= 10)]
    b = hist[(hist["AÑO"] == anio_objetivo - 1) & (hist["MES_NUM"] <= 10)]
    meses = sorted(set(a.MES_NUM) & set(b.MES_NUM))
    if len(meses) < 6:
        return 1.0
    va = a[a.MES_NUM.isin(meses)].VENTA.sum()
    vb = b[b.MES_NUM.isin(meses)].VENTA.sum()
    return float(np.clip(va / vb, 0.80, 1.40)) if vb > 0 else 1.0


def ajustar_estacional(pred_base, hist, periodo, peso_forecast=0.70, umbral=1.10):
    mes, ano = periodo % 100, periodo // 100
    if mes not in [11, 12]:
        return float(pred_base), np.nan, 1.0, False
    idx = indice_estacional(hist, ano)
    mismos = hist[(hist.MES_NUM == mes) & (hist.AÑO < ano)].sort_values("AÑO").tail(2)
    if mismos.empty or idx < umbral:
        return float(pred_base), np.nan, idx, False
    valores = mismos.VENTA.astype(float).to_numpy()
    ref_hist = valores[-1] if len(valores) == 1 else .65 * valores[-1] + .35 * valores[-2]
    crec = crecimiento_ytd(hist, ano)
    ref_ytd = ref_hist * crec
    ajustada = peso_forecast * pred_base + (1 - peso_forecast) * ref_ytd
    # Piso prudente, nunca traslada todo el crecimiento YTD.
    piso = valores[-1] * (1 + .35 * max(crec - 1, 0))
    return max(float(ajustada), float(piso)), float(ref_ytd), idx, True


def factor_antisesgo_regularizado(bt, intensidad=.60, tope=1.10, prior=30):
    """Aprende factores por mes/categoria, contraidos hacia el global."""
    if bt.empty:
        return {"global": 1.0, "mes": {}, "cat_mes": {}}
    real, pred = bt.Actual.sum(), bt.Pred_estacional.sum()
    fg = np.clip(real / pred if pred > 0 else 1.0, .95, tope)
    meses, cats = {}, {}
    for mes, g in bt.groupby(bt.Periodo % 100):
        raw = g.Actual.sum() / g.Pred_estacional.sum() if g.Pred_estacional.sum() > 0 else 1.0
        f = (len(g) * raw + prior * fg) / (len(g) + prior)
        meses[int(mes)] = float(np.clip(f, .95, tope))
    for (cat, mes), g in bt.groupby(["CATEGORIA", bt.Periodo % 100]):
        base = meses.get(int(mes), fg)
        raw = g.Actual.sum() / g.Pred_estacional.sum() if g.Pred_estacional.sum() > 0 else 1.0
        f = (len(g) * raw + prior * base) / (len(g) + prior)
        cats[(str(cat), int(mes))] = float(np.clip(f, .95, tope))
    # Se aplica solo una fraccion del factor aprendido.
    def parcial(f): return 1 + intensidad * (f - 1)
    return {"global": parcial(fg), "mes": {k: parcial(v) for k,v in meses.items()},
            "cat_mes": {k: parcial(v) for k,v in cats.items()}}


def aplicar_antisesgo(pred, categoria, periodo, calibrador):
    mes = periodo % 100
    f = calibrador.get("cat_mes", {}).get((str(categoria), mes),
        calibrador.get("mes", {}).get(mes, calibrador.get("global", 1.0)))
    return max(0.0, float(pred) * float(f)), float(f)


def backtest_integrado(df, periodos, peso_estructural=.15, peso_estacional=.70,
                       umbral=1.10, calibrador=None):
    corte = min(periodos)
    train = df[df.Periodo < corte].copy()
    real = df[df.Periodo.isin(periodos)].copy()
    if train.empty or real.empty:
        return pd.DataFrame()
    modelos = entrenar_tres_modelos(train)
    filas=[]
    for codigo in sorted(set(train.CODIGO) & set(real.CODIGO)):
        h=train[train.CODIGO==codigo].sort_values("Periodo")
        r=real[real.CODIGO==codigo].sort_values("Periodo")
        ps=r.Periodo.astype(int).tolist()
        if not ps: continue
        pv,pu,pp,pe,_=pronosticar_sku(modelos,h,ps,1-peso_estructural)
        for (_,rr),vd,u,pr,ve in zip(r.iterrows(),pv,pu,pp,pe):
            pb=(1-peso_estructural)*vd + peso_estructural*ve
            pa,ref,idx,ap=ajustar_estacional(pb,h,int(rr.Periodo),peso_estacional,umbral)
            pf,factor=(pa,1.0) if calibrador is None else aplicar_antisesgo(pa,rr.CATEGORIA,int(rr.Periodo),calibrador)
            filas.append({"CODIGO":codigo,"DESCRIPCION":rr.get("DESCRIPCION", ""),
                "CATEGORIA":rr.CATEGORIA,"COMBOS":int(rr.get("COMBOS", 0)),
                "Periodo":int(rr.Periodo),
                "Actual":float(rr.VENTA),"Venta_directa":vd,"Unidades_pred":u,"Precio_pred":pr,
                "Venta_estructural":ve,"Pred_base_hibrida":pb,"Referencia_estacional":ref,
                "Indice_estacional":idx,"Ajuste_aplicado":ap,"Pred_estacional":pa,
                "Factor_antisesgo":factor,"Prediccion":pf})
    return pd.DataFrame(filas)


def optimizar_peso_estructural(df, peso_estacional=.70, umbral=1.10):
    filas=[]; detalles={}
    for pe in np.arange(0, .31, .05):
        d=backtest_integrado(df,[202511,202512],float(pe),peso_estacional,umbral,None)
        if d.empty: continue
        m=metricas(d.Actual,d.Pred_estacional)
        score=m["WAPE"] + .00002*m["RMSE"] + .20*abs(m["SESGO"])
        filas.append({"Peso_estructural":pe,"Peso_venta_directa":1-pe,**m,"Score":score})
        detalles[round(float(pe),2)]=d
    t=pd.DataFrame(filas).sort_values(["Score","WAPE","RMSE"])
    return t, detalles



def metricas_por_categoria(datos, columna_pred="Pred_estacional", excluir_combos=False):
    """Calcula metricas correctamente sobre registros SKU-mes por categoria."""
    d = datos.copy()
    if excluir_combos:
        d = d[pd.to_numeric(d["COMBOS"], errors="coerce").fillna(0).astype(int) == 0]
    filas = []
    for categoria, g in d.groupby("CATEGORIA", dropna=False):
        m = metricas(g["Actual"], g[columna_pred])
        filas.append({
            "CATEGORIA": str(categoria),
            "SKU": int(g["CODIGO"].nunique()),
            "Registros": int(len(g)),
            "Venta_real": float(g["Actual"].sum()),
            "Pronostico": float(g[columna_pred].sum()),
            **m,
        })
    if not filas:
        return pd.DataFrame()
    return pd.DataFrame(filas).sort_values("Venta_real", ascending=False).reset_index(drop=True)


def comparacion_segmento(datos, excluir_combos=False):
    d = datos.copy()
    if excluir_combos:
        d = d[pd.to_numeric(d["COMBOS"], errors="coerce").fillna(0).astype(int) == 0]
    modelos = [
        ("Venta directa", "Venta_directa"),
        ("Hibrido sin ajuste", "Pred_base_hibrida"),
        ("Hibrido + estacional", "Pred_estacional"),
    ]
    filas = []
    for nombre, col in modelos:
        m = metricas(d["Actual"], d[col])
        filas.append({"Modelo": nombre, "SKU": int(d.CODIGO.nunique()),
                      "Venta_real": float(d.Actual.sum()),
                      "Pronostico": float(d[col].sum()), **m})
    return pd.DataFrame(filas)


def mostrar_tabla_metricas(tabla):
    if tabla.empty:
        st.warning("No hay registros para este segmento.")
        return
    formato = {
        "Venta_real": "${:,.2f}", "Pronostico": "${:,.2f}",
        "WAPE": "{:.2f}%", "MAE": "${:,.2f}", "RMSE": "${:,.2f}",
        "R2": "{:.3f}", "SESGO": "{:+.2f}%",
    }
    st.dataframe(tabla.style.format(formato), use_container_width=True, hide_index=True)



def separar_categorias_sin_combos(datos, columna_valor="Prediccion_final"):
    """Excluye combos y devuelve un diccionario con un DataFrame por categoria."""
    d = datos.copy()
    d["COMBOS"] = pd.to_numeric(d.get("COMBOS", 0), errors="coerce").fillna(0).astype(int)
    d = d[d["COMBOS"] == 0].copy()
    salida = {}
    for categoria, g in d.groupby("CATEGORIA", dropna=False):
        nombre = str(categoria).strip() or "SIN_CATEGORIA"
        salida[nombre] = g.sort_values(["Periodo", "CODIGO"]).reset_index(drop=True)
    return salida


def nombre_archivo_seguro(texto):
    import re
    nombre = re.sub(r"[^A-Za-z0-9_-]+", "_", str(texto).strip())
    return nombre.strip("_") or "SIN_CATEGORIA"

def forecast_2026_integrado(df, peso_estructural, peso_estacional, umbral, calibrador):
    modelos=entrenar_tres_modelos(df); filas=[]
    for codigo,h in df.groupby("CODIGO"):
        h=h.sort_values("Periodo"); meta=h.iloc[-1]; periodos=[202611,202612]
        pv,pu,pp,pe,_=pronosticar_sku(modelos,h,periodos,1-peso_estructural)
        for periodo,vd,u,pr,ve in zip(periodos,pv,pu,pp,pe):
            pb=(1-peso_estructural)*vd+peso_estructural*ve
            pa,ref,idx,ap=ajustar_estacional(pb,h,periodo,peso_estacional,umbral)
            pf,factor=aplicar_antisesgo(pa,meta.CATEGORIA,periodo,calibrador)
            filas.append({"CODIGO":codigo,"DESCRIPCION":meta.DESCRIPCION,"CATEGORIA":meta.CATEGORIA,
                "COMBOS":int(meta.get("COMBOS", 0)),
                "Periodo":periodo,"Venta_directa":vd,"Unidades_pred":u,"Precio_pred":pr,
                "Venta_estructural":ve,"Pred_base_hibrida":pb,"Referencia_estacional":ref,
                "Indice_estacional":idx,"Ajuste_aplicado":ap,"Pred_estacional":pa,
                "Factor_antisesgo":factor,"Prediccion_final":pf})
    return pd.DataFrame(filas)



def optimizar_configuracion_modelo(df):
    """Búsqueda secuencial robusta para Nov-Dic 2025.

    Etapa 1: peso estructural 0%-30%.
    Etapa 2: peso forecast estacional y umbral.
    Etapa 3: recomienda antisesgo conservador sin usarlo para inflar la métrica 2025.
    """
    resultados_etapa1 = []
    detalles_etapa1 = {}
    for peso_estructural in np.arange(0.00, 0.31, 0.05):
        bt = backtest_integrado(
            df, [202511, 202512], float(peso_estructural), 0.70, 1.10, None
        )
        if bt.empty:
            continue
        bt_sc = bt[bt["COMBOS"].astype(int) == 0].copy()
        m = metricas(bt_sc.Actual, bt_sc.Pred_estacional)
        score = m["WAPE"] + 0.00002 * m["RMSE"] + 0.25 * abs(m["SESGO"])
        resultados_etapa1.append({
            "Peso_estructural": float(peso_estructural), **m, "Score": score
        })
        detalles_etapa1[round(float(peso_estructural), 2)] = bt
    tabla1 = pd.DataFrame(resultados_etapa1).sort_values("Score")
    if tabla1.empty:
        return pd.DataFrame(), None, None
    mejor_pe = float(tabla1.iloc[0]["Peso_estructural"])

    resultados_etapa2 = []
    detalles_etapa2 = {}
    for peso_estacional in [0.55, 0.65, 0.70, 0.75, 0.85]:
        for umbral in [1.00, 1.05, 1.10, 1.15, 1.20]:
            bt = backtest_integrado(
                df, [202511, 202512], mejor_pe,
                float(peso_estacional), float(umbral), None
            )
            if bt.empty:
                continue
            bt_sc = bt[bt["COMBOS"].astype(int) == 0].copy()
            m = metricas(bt_sc.Actual, bt_sc.Pred_estacional)
            # Penaliza sesgo y RMSE para no elegir una configuración que solo minimice WAPE.
            score = m["WAPE"] + 0.00002 * m["RMSE"] + 0.25 * abs(m["SESGO"])
            clave = (float(peso_estacional), float(umbral))
            resultados_etapa2.append({
                "Peso_estructural": mejor_pe,
                "Peso_forecast_estacional": clave[0],
                "Umbral_estacional": clave[1],
                **m, "Score": score,
            })
            detalles_etapa2[clave] = bt
    tabla2 = pd.DataFrame(resultados_etapa2).sort_values("Score")
    if tabla2.empty:
        return tabla1, None, None

    ganador = tabla2.iloc[0].to_dict()
    clave = (ganador["Peso_forecast_estacional"], ganador["Umbral_estacional"])
    bt_ganador = detalles_etapa2[clave]
    sesgo = float(ganador["SESGO"])

    # Antisesgo se recomienda, pero no se optimiza contra el mismo 2025 para evitar fuga.
    if sesgo <= -10:
        intensidad, tope = 0.70, 1.10
    elif sesgo <= -5:
        intensidad, tope = 0.60, 1.08
    elif sesgo < -2:
        intensidad, tope = 0.45, 1.06
    elif sesgo >= 5:
        intensidad, tope = 0.35, 1.04
    else:
        intensidad, tope = 0.25, 1.04
    ganador["Intensidad_antisesgo"] = intensidad
    ganador["Tope_antisesgo"] = tope
    return tabla2.reset_index(drop=True), ganador, bt_ganador

def grafica_ventas_mensuales(df_historico, pronostico=None, categoria="TODAS",
                             excluir_combos=True, codigo="TODOS"):
    """Muestra ventas mensuales reales y forecast final Nov-Dic 2026."""
    h = df_historico.copy()
    if "COMBOS" not in h.columns:
        h["COMBOS"] = 0
    h["COMBOS"] = pd.to_numeric(h["COMBOS"], errors="coerce").fillna(0).astype(int)

    if excluir_combos:
        h = h[h["COMBOS"] == 0]
    if categoria and categoria != "TODAS":
        h = h[h["CATEGORIA"].astype(str).str.strip() == str(categoria).strip()]
    if codigo and codigo != "TODOS":
        h = h[h["CODIGO"].astype(str) == str(codigo)]

    hist = h.groupby(["AÑO", "MES_NUM"], as_index=False)["VENTA"].sum()
    hist["Fecha"] = pd.to_datetime(dict(
        year=hist["AÑO"].astype(int), month=hist["MES_NUM"].astype(int), day=1
    ))
    hist = hist.rename(columns={"VENTA": "Valor"})
    hist["Serie"] = "Venta real"
    partes = [hist[["Fecha", "Valor", "Serie"]]]

    if pronostico is not None and not pronostico.empty:
        futuro = pronostico.copy()
        if "COMBOS" not in futuro.columns:
            futuro["COMBOS"] = 0
        futuro["COMBOS"] = pd.to_numeric(futuro["COMBOS"], errors="coerce").fillna(0).astype(int)
        if excluir_combos:
            futuro = futuro[futuro["COMBOS"] == 0]
        if categoria and categoria != "TODAS":
            futuro = futuro[futuro["CATEGORIA"].astype(str).str.strip() == str(categoria).strip()]
        if codigo and codigo != "TODOS":
            futuro = futuro[futuro["CODIGO"].astype(str) == str(codigo)]
        if not futuro.empty:
            futuro = futuro.groupby("Periodo", as_index=False)["Prediccion_final"].sum()
            futuro["Fecha"] = pd.to_datetime(dict(
                year=(futuro["Periodo"] // 100).astype(int),
                month=(futuro["Periodo"] % 100).astype(int), day=1
            ))
            futuro = futuro.rename(columns={"Prediccion_final": "Valor"})
            futuro["Serie"] = "Forecast 2026"
            partes.append(futuro[["Fecha", "Valor", "Serie"]])

    datos = pd.concat(partes, ignore_index=True).sort_values("Fecha")
    if datos.empty:
        st.warning("No hay datos para la selección actual.")
        return

    grafica = alt.Chart(datos).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X("Fecha:T", title="Mes", axis=alt.Axis(format="%b %Y", labelAngle=-45)),
        y=alt.Y("Valor:Q", title="Venta", axis=alt.Axis(format="$,.2s")),
        color=alt.Color("Serie:N", title="Serie",
            scale=alt.Scale(domain=["Venta real", "Forecast 2026"],
                            range=["#2F6BFF", "#F59E0B"])),
        tooltip=[
            alt.Tooltip("Fecha:T", title="Mes", format="%B %Y"),
            alt.Tooltip("Serie:N", title="Serie"),
            alt.Tooltip("Valor:Q", title="Venta", format="$,.2f"),
        ],
    ).properties(height=430).interactive()
    st.altair_chart(grafica, use_container_width=True)


archivos = st.file_uploader("Carga VENTAS 2024, 2025 y 2026", type=["xlsx"], accept_multiple_files=True)
with st.sidebar:
    st.header("Modelo integrado")
    peso_estacional = st.slider("Peso forecast frente a referencia estacional", .50, .90, .70, .05)
    umbral = st.slider("Umbral estacional Nov-Dic", 1.00, 1.50, 1.10, .05)
    intensidad = st.slider("Intensidad antisesgo", 0.0, 1.0, .60, .05)
    tope = st.slider("Tope factor antisesgo", 1.02, 1.20, 1.10, .01)

if archivos:
    try:
        df, diagnostico = cargar_archivos([a.getvalue() for a in archivos], [a.name for a in archivos])
        st.success(f"{len(df):,} registros | {df.CODIGO.nunique():,} productos | años {sorted(df.AÑO.unique())}")
        with st.expander("Diagnóstico de carga"):
            st.dataframe(diagnostico, use_container_width=True, hide_index=True)

        st.subheader("Evolución mensual de ventas")
        categorias_grafica = ["TODAS"] + sorted(df["CATEGORIA"].astype(str).str.strip().unique().tolist())
        col_g1, col_g2, col_g3 = st.columns([2, 2, 1])
        with col_g1:
            categoria_grafica = st.selectbox("Categoría", categorias_grafica, key="categoria_grafica_mensual")
        df_codigos_grafica = df.copy()
        if categoria_grafica != "TODAS":
            df_codigos_grafica = df_codigos_grafica[df_codigos_grafica["CATEGORIA"].astype(str).str.strip() == categoria_grafica]
        with col_g2:
            codigo_grafica = st.selectbox("Producto", ["TODOS"] + sorted(df_codigos_grafica["CODIGO"].astype(str).unique().tolist()), key="codigo_grafica_mensual")
        with col_g3:
            quitar_combos_grafica = st.checkbox("Excluir combos", value=True, key="quitar_combos_grafica")
        pron_grafica = st.session_state.get("pron_final", pd.DataFrame())
        grafica_ventas_mensuales(df, pronostico=pron_grafica, categoria=categoria_grafica,
                                 excluir_combos=quitar_combos_grafica, codigo=codigo_grafica)
        if pron_grafica.empty:
            st.caption("Histórico mensual 2024-2026. Al generar el forecast aparecerán noviembre y diciembre de 2026 en naranja.")
        else:
            st.caption("Azul: venta real mensual. Naranja: forecast final noviembre-diciembre de 2026.")

        st.subheader("1. Búsqueda de la mejor configuración")
        st.caption(
            "Optimiza con Nov-Dic 2025 y productos sin combos: peso unidades × precio, "
            "peso frente a referencia estacional y umbral. La intensidad antisesgo se "
            "recomienda de forma conservadora para evitar sobreajustar 2025."
        )
        if st.button("Buscar mejor configuración completa", type="primary"):
            with st.spinner("Evaluando configuraciones. Puede tardar varios minutos..."):
                tabla_config, mejor_config, bt_mejor_config = optimizar_configuracion_modelo(df)
            st.session_state["tabla_config_completa"] = tabla_config
            st.session_state["mejor_config"] = mejor_config
            st.session_state["bt_mejor_config"] = bt_mejor_config

        if st.session_state.get("mejor_config") is not None:
            mejor_config = st.session_state["mejor_config"]
            tabla_config = st.session_state["tabla_config_completa"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Venta directa", f"{1-mejor_config['Peso_estructural']:.0%}")
            c2.metric("Unidades × precio", f"{mejor_config['Peso_estructural']:.0%}")
            c3.metric("Peso forecast", f"{mejor_config['Peso_forecast_estacional']:.0%}")
            c4.metric("Umbral estacional", f"{mejor_config['Umbral_estacional']:.2f}")
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("WAPE sin combos", f"{mejor_config['WAPE']:.2f}%")
            c6.metric("RMSE", f"${mejor_config['RMSE']:,.0f}")
            c7.metric("Sesgo", f"{mejor_config['SESGO']:+.2f}%")
            c8.metric("R²", f"{mejor_config['R2']:.3f}")
            st.success(
                f"Configuración recomendada: peso forecast {mejor_config['Peso_forecast_estacional']:.2f}, "
                f"umbral {mejor_config['Umbral_estacional']:.2f}, intensidad antisesgo "
                f"{mejor_config['Intensidad_antisesgo']:.2f} y tope {mejor_config['Tope_antisesgo']:.2f}."
            )
            st.dataframe(
                tabla_config.head(10).style.format({
                    "Peso_estructural":"{:.0%}", "Peso_forecast_estacional":"{:.0%}",
                    "Umbral_estacional":"{:.2f}", "WAPE":"{:.2f}%",
                    "MAE":"${:,.2f}", "RMSE":"${:,.2f}", "R2":"{:.3f}",
                    "SESGO":"{:+.2f}%", "Score":"{:.3f}"
                }), use_container_width=True, hide_index=True
            )
            st.download_button(
                "Descargar ranking de configuraciones",
                tabla_config.to_csv(index=False).encode("utf-8-sig"),
                "ranking_configuraciones_nov_dic_2025.csv", "text/csv"
            )

        st.divider()
        st.subheader("2. Optimización rápida del peso unidades × precio")
        st.caption("Busca de 0% a 30%. Nov-Dic 2025 se usa como validación, no como entrenamiento.")
        if st.button("Optimizar modelo integrado", type="primary"):
            with st.spinner("Entrenando y comparando pesos..."):
                tabla, detalles = optimizar_peso_estructural(df,peso_estacional,umbral)
            st.session_state["tabla_pesos"] = tabla
            st.session_state["detalles_pesos"] = detalles
        if "tabla_pesos" in st.session_state:
            tabla=st.session_state["tabla_pesos"]; detalles=st.session_state["detalles_pesos"]
            st.dataframe(tabla.style.format({"Peso_estructural":"{:.0%}","Peso_venta_directa":"{:.0%}",
                "WAPE":"{:.2f}%","MAE":"${:,.2f}","RMSE":"${:,.2f}","R2":"{:.3f}","SESGO":"{:+.2f}%"}),
                use_container_width=True,hide_index=True)
            mejor=float(tabla.iloc[0].Peso_estructural)
            peso_estacional_activo = peso_estacional
            umbral_activo = umbral
            intensidad_activa = intensidad
            tope_activo = tope
            if st.session_state.get("mejor_config") is not None:
                cfg = st.session_state["mejor_config"]
                mejor = float(cfg["Peso_estructural"])
                peso_estacional_activo = float(cfg["Peso_forecast_estacional"])
                umbral_activo = float(cfg["Umbral_estacional"])
                intensidad_activa = float(cfg["Intensidad_antisesgo"])
                tope_activo = float(cfg["Tope_antisesgo"])
                bt_base = st.session_state["bt_mejor_config"]
                st.success("Se utilizará la configuración completa optimizada para generar el forecast 2026.")
            else:
                bt_base=detalles[round(mejor,2)]
            st.success(f"Peso recomendado: {1-mejor:.0%} venta directa + {mejor:.0%} unidades × precio")

            calibrador=factor_antisesgo_regularizado(bt_base,intensidad_activa,tope_activo,prior=30)
            bt_cal=bt_base.copy()
            ajustes=[aplicar_antisesgo(r.Pred_estacional,r.CATEGORIA,int(r.Periodo),calibrador) for _,r in bt_cal.iterrows()]
            bt_cal["Prediccion"]=[x[0] for x in ajustes]
            # No se presenta esta metrica como validacion independiente, porque usa 2025 para calibrar.
            st.info(f"Factor global antisesgo parcial: {calibrador['global']:.3f}. Se reservará para el forecast 2026.")

            st.subheader("2. Comparacion honesta Nov-Dic 2025")
            tab_total, tab_sin_combo, tab_categoria, tab_cat_sin_combo = st.tabs([
                "Total", "Total sin combos", "Por categoria", "Por categoria sin combos"
            ])

            with tab_total:
                comp_total = comparacion_segmento(bt_base, excluir_combos=False)
                mostrar_tabla_metricas(comp_total)

            with tab_sin_combo:
                comp_sin = comparacion_segmento(bt_base, excluir_combos=True)
                mostrar_tabla_metricas(comp_sin)
                eliminados = bt_base[bt_base["COMBOS"].astype(int) == 1]["CODIGO"].nunique()
                st.caption(f"Se excluyeron {eliminados:,} SKU identificados como combo.")

            with tab_categoria:
                modelo_cat = st.selectbox(
                    "Modelo para metricas por categoria",
                    ["Hibrido + estacional", "Venta directa", "Hibrido sin ajuste"],
                    key="modelo_categoria_total",
                )
                mapa_col = {"Hibrido + estacional":"Pred_estacional",
                            "Venta directa":"Venta_directa",
                            "Hibrido sin ajuste":"Pred_base_hibrida"}
                cat_total = metricas_por_categoria(bt_base, mapa_col[modelo_cat], False)
                mostrar_tabla_metricas(cat_total)
                st.download_button("Descargar metricas por categoria", cat_total.to_csv(index=False).encode("utf-8-sig"),
                                   "metricas_categoria_nov_dic_2025.csv", "text/csv")

            with tab_cat_sin_combo:
                modelo_cat_sc = st.selectbox(
                    "Modelo para metricas por categoria sin combos",
                    ["Hibrido + estacional", "Venta directa", "Hibrido sin ajuste"],
                    key="modelo_categoria_sin_combo",
                )
                mapa_col_sc = {"Hibrido + estacional":"Pred_estacional",
                               "Venta directa":"Venta_directa",
                               "Hibrido sin ajuste":"Pred_base_hibrida"}
                cat_sin = metricas_por_categoria(bt_base, mapa_col_sc[modelo_cat_sc], True)
                mostrar_tabla_metricas(cat_sin)
                st.download_button("Descargar categorias sin combos", cat_sin.to_csv(index=False).encode("utf-8-sig"),
                                   "metricas_categoria_sin_combos_nov_dic_2025.csv", "text/csv")

            st.caption("WAPE se calcula como suma del error absoluto dividida entre la venta real de cada categoria. El factor antisesgo no se presenta como mejora de 2025 porque fue aprendido con ese mismo periodo.")

            st.subheader("3. Forecast noviembre-diciembre 2026")
            if st.button("Generar forecast final 2026", type="primary"):
                with st.spinner("Entrenando tres modelos y aplicando ajustes..."):
                    pron=forecast_2026_integrado(df,mejor,peso_estacional_activo,umbral_activo,calibrador)
                st.session_state["pron_final"]=pron
            if "pron_final" in st.session_state:
                pron=st.session_state["pron_final"]
                st.dataframe(pron,use_container_width=True,hide_index=True)
                st.download_button("Descargar forecast final CSV",pron.to_csv(index=False).encode("utf-8-sig"),
                    "forecast_integrado_nov_dic_2026.csv","text/csv")

                st.subheader("4. Categorias separadas sin combos")
                categorias_sin_combo = separar_categorias_sin_combos(pron)
                if not categorias_sin_combo:
                    st.warning("No quedaron productos despues de excluir COMBOS = 1.")
                else:
                    categorias_disponibles = sorted(categorias_sin_combo)
                    categoria_sel = st.selectbox(
                        "Selecciona una categoria sin combos",
                        categorias_disponibles,
                        key="categoria_forecast_sin_combo",
                    )
                    detalle_categoria = categorias_sin_combo[categoria_sel]
                    resumen_categoria = detalle_categoria.groupby(
                        ["CATEGORIA", "Periodo"], as_index=False
                    ).agg(
                        SKU=("CODIGO", "nunique"),
                        Unidades_pred=("Unidades_pred", "sum"),
                        Venta_directa=("Venta_directa", "sum"),
                        Venta_estructural=("Venta_estructural", "sum"),
                        Pred_estacional=("Pred_estacional", "sum"),
                        Prediccion_final=("Prediccion_final", "sum"),
                    )
                    st.markdown(f"#### {categoria_sel}")
                    st.markdown("##### Ventas mensuales históricas y forecast")
                    grafica_ventas_mensuales(df, pronostico=pron, categoria=categoria_sel,
                                             excluir_combos=True, codigo="TODOS")
                    st.dataframe(
                        resumen_categoria.style.format({
                            "Unidades_pred": "{:,.0f}",
                            "Venta_directa": "${:,.2f}",
                            "Venta_estructural": "${:,.2f}",
                            "Pred_estacional": "${:,.2f}",
                            "Prediccion_final": "${:,.2f}",
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.dataframe(detalle_categoria, use_container_width=True, hide_index=True)
                    archivo_categoria = f"forecast_{nombre_archivo_seguro(categoria_sel)}_sin_combos.csv"
                    st.download_button(
                        "Descargar categoria seleccionada sin combos",
                        detalle_categoria.to_csv(index=False).encode("utf-8-sig"),
                        archivo_categoria,
                        "text/csv",
                    )

                    # ZIP con un CSV independiente por categoria, todos sin combos.
                    import zipfile
                    from io import BytesIO
                    memoria_zip = BytesIO()
                    with zipfile.ZipFile(memoria_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                        for categoria, tabla_cat in categorias_sin_combo.items():
                            zf.writestr(
                                f"forecast_{nombre_archivo_seguro(categoria)}_sin_combos.csv",
                                tabla_cat.to_csv(index=False).encode("utf-8-sig"),
                            )
                    st.download_button(
                        "Descargar todas las categorias separadas en ZIP",
                        memoria_zip.getvalue(),
                        "forecast_categorias_sin_combos_2026.zip",
                        "application/zip",
                    )
    except Exception as e:
        st.exception(e)
else:
    st.info("Carga los tres archivos Excel para comenzar.")

