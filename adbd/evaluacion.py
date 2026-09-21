"""
Evaluación predictiva. Todo se mide sobre el conjunto COMPLETO sin submuestrear.

Por qué AUC-PR y no accuracy: con 5,14% de positivos, predecir siempre "no click"
da 94,86% de accuracy y cero valor. AUC-PR se mueve con la prevalencia, así que
medirlo sobre una muestra con 35% de positivos daría un número que no existe en
producción. AUC-ROC sí es invariante al submuestreo de negativos (es puro orden),
y esa diferencia entre las dos métricas se reporta explícitamente porque es la
prueba de que la recalibración se hizo bien.

Y una métrica de negocio, porque el panel de la defensa no compra un AUC: el
**lift del decil superior**. En una plataforma de display no se decide "click o
no click", se ordena inventario; lo que importa es cuánto mejor rinde el 10% de
impresiones que el modelo pone arriba comparado con servir al azar.
"""
from __future__ import annotations

import math

import pandas as pd
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as F

from . import config
from .gold import OBJETIVO

EPS = 1e-12


def metricas_binarias(pred, col_p: str = "p_calibrada") -> dict:
    """AUC-ROC, AUC-PR, LogLoss, Brier y el CTR medio predicho vs el observado.

    Las dos últimas columnas son el control de calibración: si el CTR medio
    predicho no se parece al observado, la probabilidad no es una probabilidad
    y cualquier umbral de negocio construido sobre ella está mal puesto.
    """
    p = F.col(col_p)
    y = F.col(OBJETIVO).cast("double")
    pc = F.least(F.greatest(p, F.lit(EPS)), F.lit(1 - EPS))

    agg = pred.select(
        F.count(F.lit(1)).alias("n"),
        F.avg(y).alias("ctr_observado"),
        F.avg(p).alias("ctr_predicho"),
        F.avg(-(y * F.log(pc) + (1 - y) * F.log(1 - pc))).alias("logloss"),
        F.avg(F.pow(p - y, 2)).alias("brier"),
    ).collect()[0].asDict()

    ev = BinaryClassificationEvaluator(labelCol=OBJETIVO, rawPredictionCol=col_p)
    agg["auc_roc"] = ev.setMetricName("areaUnderROC").evaluate(pred)
    agg["auc_pr"] = ev.setMetricName("areaUnderPR").evaluate(pred)
    agg["lift_auc_pr"] = agg["auc_pr"] / agg["ctr_observado"] if agg["ctr_observado"] else None
    agg["sesgo_calibracion"] = (
        agg["ctr_predicho"] / agg["ctr_observado"] if agg["ctr_observado"] else None
    )
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in agg.items()}


def tabla_deciles(pred, col_p: str = "p_calibrada", n: int = 10) -> pd.DataFrame:
    """CTR observado por decil de score. Es la traducción a negocio del AUC.

    Los cortes se obtienen con approxQuantile (distribuido) y no con NTILE: una
    ventana sin PARTITION BY sobre millones de filas colapsa todo a una sola
    partición y convierte la evaluación en el paso más caro del notebook.
    """
    cortes = pred.approxQuantile(col_p, [i / n for i in range(1, n)], 0.001)
    cortes = sorted(set(cortes))
    expr = F.lit(0)
    for c in cortes:
        expr = expr + (F.col(col_p) > F.lit(c)).cast("int")
    d = (pred.withColumn("decil", expr)
             .groupBy("decil")
             .agg(F.count(F.lit(1)).alias("impresiones"),
                  F.sum(OBJETIVO).alias("clicks"),
                  F.avg(col_p).alias("p_media"))
             .orderBy(F.col("decil").desc())
             .toPandas())
    global_ctr = d.clicks.sum() / d.impresiones.sum()
    d["ctr_pct"] = (d.clicks / d.impresiones * 100).round(3)
    d["p_media_pct"] = (d.p_media * 100).round(3)
    d["lift"] = (d.clicks / d.impresiones / global_ctr).round(3)
    d["decil"] = d.decil.map(lambda i: f"D{n - i}")   # D1 = el 10% de mayor score
    return d[["decil", "impresiones", "clicks", "ctr_pct", "p_media_pct", "lift"]]


def curva_pr(pred, col_p: str = "p_calibrada", puntos: int = 60) -> pd.DataFrame:
    """Precisión y recall en `puntos` umbrales. Se calcula agregando por bin de
    score en una sola pasada: recorrer la tabla una vez por umbral serían 60
    recorridos completos sobre millones de filas."""
    bins = (pred.withColumn("bin", F.least(
                F.floor(F.col(col_p) * puntos), F.lit(puntos - 1)).cast("int"))
                .groupBy("bin")
                .agg(F.count(F.lit(1)).alias("n"), F.sum(OBJETIVO).alias("pos"))
                .orderBy(F.col("bin").desc())
                .toPandas())
    bins["n_acum"] = bins.n.cumsum()
    bins["pos_acum"] = bins.pos.cumsum()
    total_pos = bins.pos.sum()
    bins["precision"] = bins.pos_acum / bins.n_acum
    bins["recall"] = bins.pos_acum / total_pos if total_pos else 0.0
    bins["umbral"] = bins.bin / puntos
    return bins[["umbral", "n_acum", "precision", "recall"]]


def comparar_submuestreo(pred_completo, pred_muestra, col_p: str = "p_calibrada") -> pd.DataFrame:
    """La evidencia de que evaluar sobre la muestra engaña: AUC-ROC coincide
    (es orden puro) y AUC-PR no (depende de la prevalencia)."""
    a, b = metricas_binarias(pred_completo, col_p), metricas_binarias(pred_muestra, col_p)
    filas = []
    for k in ("ctr_observado", "auc_roc", "auc_pr", "logloss"):
        filas.append({"metrica": k, "conjunto_completo": a[k], "submuestra": b[k],
                      "diferencia_rel": round((b[k] - a[k]) / a[k], 4) if a[k] else None})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Ranking (ALS)
# ---------------------------------------------------------------------------
def metricas_ranking(recomendadas, reales, k: int = None) -> dict:
    """Precision@k, Recall@k, MAP@k y NDCG@k.

    `recomendadas` y `reales`: DataFrames con (userid, lista). Se evalúa solo
    sobre usuarios presentes en ambos; los usuarios frío —sin historia en train—
    se cuentan aparte y se reportan, porque esconderlos infla todas las métricas.
    """
    k = k or config.TOP_K
    j = recomendadas.join(reales, on="userid", how="inner").collect()
    n_rec = recomendadas.count()
    n_real = reales.count()

    p, r, ap, ndcg = [], [], [], []
    for fila in j:
        rec = list(fila["recomendadas"])[:k]
        real = set(fila["reales"])
        if not real:
            continue
        aciertos = [1 if i in real else 0 for i in rec]
        n_ac = sum(aciertos)
        p.append(n_ac / k)
        r.append(n_ac / len(real))
        acum, prec = 0, 0.0
        for i, a in enumerate(aciertos, start=1):
            if a:
                acum += 1
                prec += acum / i
        ap.append(prec / min(len(real), k))
        dcg = sum(a / math.log2(i + 1) for i, a in enumerate(aciertos, start=1))
        idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(real), k) + 1))
        ndcg.append(dcg / idcg if idcg else 0.0)

    n = len(p)
    return {
        "k": k,
        "usuarios_evaluados": n,
        "usuarios_con_verdad": n_real,
        "usuarios_con_recomendacion": n_rec,
        "cobertura_pct": round(100 * n / n_real, 2) if n_real else 0.0,
        "precision_at_k": round(sum(p) / n, 6) if n else 0.0,
        "recall_at_k": round(sum(r) / n, 6) if n else 0.0,
        "map_at_k": round(sum(ap) / n, 6) if n else 0.0,
        "ndcg_at_k": round(sum(ndcg) / n, 6) if n else 0.0,
    }


# ---------------------------------------------------------------------------
# Gráficos (artefactos de MLflow)
# ---------------------------------------------------------------------------
def graficar_pr(curvas: dict, prevalencia: float, ruta: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=130)
    for nombre, d in curvas.items():
        ax.plot(d["recall"], d["precision"], label=nombre, lw=1.6)
    ax.axhline(prevalencia, ls="--", lw=1, color="#888",
               label=f"azar (prevalencia {prevalencia*100:.2f}%)")
    ax.set_xlabel("recall"); ax.set_ylabel("precisión")
    ax.set_title("Curva precisión-recall · día de test completo")
    ax.legend(fontsize=7); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(ruta); plt.close(fig)
    return ruta


def graficar_calibracion(deciles: pd.DataFrame, ruta: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=130)
    ax.plot(deciles.p_media_pct, deciles.ctr_pct, "o-", lw=1.5)
    lim = max(deciles.p_media_pct.max(), deciles.ctr_pct.max()) * 1.08
    ax.plot([0, lim], [0, lim], ls="--", lw=1, color="#888", label="calibración perfecta")
    ax.set_xlabel("CTR predicho medio (%)"); ax.set_ylabel("CTR observado (%)")
    ax.set_title("Calibración por decil de score")
    ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(ruta); plt.close(fig)
    return ruta


def graficar_curva_aprendizaje(d: pd.DataFrame, ruta: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(6, 4.2), dpi=130)
    ax1.plot(d.filas, d.auc_pr, "o-", color="#1f77b4", label="AUC-PR")
    ax1.set_xlabel("filas de entrenamiento"); ax1.set_ylabel("AUC-PR", color="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(d.filas, d.t_ajuste_s, "s--", color="#d62728", label="tiempo de ajuste (s)")
    ax2.set_ylabel("segundos", color="#d62728")
    ax1.set_title("¿Más datos, o más modelo? Curva de aprendizaje y su costo")
    ax1.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(ruta); plt.close(fig)
    return ruta
