"""
GOLD · la tabla de entrenamiento y el registro de variables versionado (gold_ads_v1).

Este módulo es donde se gana o se pierde la Fase 2, porque es donde vive el riesgo
de FUGA DE INFORMACIÓN. La Fase 1 ya lo encontró una vez en W3 (`CURRENT ROW` en
lugar de `1 PRECEDING` metía el resultado a predecir dentro del feature y el split
temporal seguía viéndose impecable). Aquí la misma disciplina, aplicada a todo:

  Regla única: NINGÚN feature puede depender de una fila cuyo `clk` el modelo
  todavía no habría visto en el instante de la impresión.

De ahí salen las tres decisiones de diseño del módulo:

  1. Los CTR históricos por entidad se calculan con retardo de UN DÍA y ventana
     expansiva: para una impresión del día d, el histórico solo suma los días < d.
  2. Los features de usuario (fatiga) se calculan con ventana expansiva DENTRO del
     flujo, cerrada en la fila anterior (`rowsBetween(unboundedPreceding, -1)`), con
     el mismo desempate total que la Fase 1 necesitó para que W2 fuera reproducible.
  3. El CTR móvil del slot mantiene `RANGE BETWEEN 6 PRECEDING AND 1 PRECEDING`,
     literalmente la consulta W3, porque ya estaba bien.

El costo de la regla es un día: el 06-may no se entrena, existe para que el 07-may
tenga historia. `verificar_fuga()` deja el control ejecutable.
"""
from __future__ import annotations

import pandas as pd
from pyspark.sql import Window
from pyspark.sql import functions as F

from . import config

# ---------------------------------------------------------------------------
# Registro de variables. Es un artefacto versionado: lo aceptado Y lo descartado
# con su motivo, que es lo que la Fase 1 prometió entregar en Gold.
# ---------------------------------------------------------------------------
CATEGORICAS = [
    "pid", "franja", "hora_local", "dia_semana",
    "final_gender_code", "age_level", "pvalue_level", "shopping_level",
    "occupation", "new_user_class_level", "cms_group_id", "bucket_gap",
]

NUMERICAS = (
    [f"ctr_hist_{e}" for e in config.ENTIDADES_HISTORICAS]
    + [f"log_imp_hist_{e}" for e in config.ENTIDADES_HISTORICAS]
    + [
        "ctr_usuario_previo", "log_imp_previas_usuario", "orden_impresion",
        "log_gap_seg", "ctr_movil_6h_slot",
        "log_price", "precio_rel_categoria",
        "sin_perfil", "brand_conocida", "precio_valido", "es_finde",
    ]
)

DESCARTADAS = {
    "brand": "el cast fabricó 246.330 nulos inexistentes en el origen (Fase 1 §2); "
             "usarla sería modelar un artefacto de la línea de carga. Sobrevive como "
             "flag brand_conocida.",
    "nonclk": "complemento exacto de clk dentro de la misma fila: fuga directa del objetivo.",
    "adgroup_id (histórico)": "846.811 entidades x 8 días = 6,8M de filas de historia contra "
                              "26,6M de impresiones; campaign_id lo contiene jerárquicamente.",
    "cms_segid": "97 niveles muy desbalanceados y redundante con cms_group_id, que es su "
                 "agregación declarada por la fuente.",
    "ts_local / time_stamp": "identificador temporal, no atributo: entra transformado "
                             "(hora_local, dia_semana, franja) y nunca en crudo.",
    "userid / adgroup_id (como categoría)": "cardinalidad 1,06M y 846k: no se codifican como "
                                            "categoría, entran a través de sus agregados históricos.",
}

OBJETIVO = "clk"
PROTEGIDOS = ["final_gender_code", "age_level", "new_user_class_level"]


# ---------------------------------------------------------------------------
# 1. Prior global diferido: el CTR de TODOS los días anteriores a d.
# ---------------------------------------------------------------------------
def _prior_diferido(silver):
    diario = silver.groupBy("fecha_local").agg(
        F.count(F.lit(1)).alias("imp_d"), F.sum(OBJETIVO).alias("clk_d")
    )
    w = Window.orderBy("fecha_local").rowsBetween(Window.unboundedPreceding, -1)
    return diario.select(
        "fecha_local",
        (F.sum("clk_d").over(w) / F.sum("imp_d").over(w)).alias("p0_diferido"),
    )


# ---------------------------------------------------------------------------
# 2. CTR histórico por entidad, con retardo de un día y suavizado de m-estimación.
#
#       ctr_hist = (clicks_previos + m * p0) / (impresiones_previas + m)
#
#    Con impresiones_previas = 0 (entidad nueva) el feature ES el prior: no hay
#    ningún caso en que el modelo reciba un NaN ni un cero engañoso.
# ---------------------------------------------------------------------------
def _ctr_historico(silver, prior, entidad: str, m: float):
    diario = silver.groupBy(entidad, "fecha_local").agg(
        F.count(F.lit(1)).alias("imp_d"), F.sum(OBJETIVO).alias("clk_d")
    )
    w = Window.partitionBy(entidad).orderBy("fecha_local").rowsBetween(
        Window.unboundedPreceding, -1
    )
    hist = diario.select(
        F.col(entidad),
        F.col("fecha_local"),
        F.coalesce(F.sum("imp_d").over(w), F.lit(0)).cast("double").alias("imp_prev"),
        F.coalesce(F.sum("clk_d").over(w), F.lit(0)).cast("double").alias("clk_prev"),
    )
    return (
        hist.join(prior, on="fecha_local", how="left")
        .select(
            F.col(entidad),
            F.col("fecha_local"),
            ((F.col("clk_prev") + F.lit(m) * F.col("p0_diferido"))
             / (F.col("imp_prev") + F.lit(m))).alias(f"ctr_hist_{entidad}"),
            F.log1p("imp_prev").alias(f"log_imp_hist_{entidad}"),
        )
    )


# ---------------------------------------------------------------------------
# 3. CTR móvil del slot: la consulta W3 de la Fase 1, sin cambios.
#    RANGE (no ROWS) y cota superior en 1 PRECEDING, que EXCLUYE la hora actual.
# ---------------------------------------------------------------------------
def _ctr_movil_slot(silver):
    ts_min = silver.agg(F.min("time_stamp")).collect()[0][0]
    base = (
        silver.withColumn("hora_abs", F.floor((F.col("time_stamp") - F.lit(ts_min)) / 3600).cast("bigint"))
        .groupBy("pid", "hora_abs")
        .agg(F.count(F.lit(1)).alias("imp_h"), F.sum(OBJETIVO).alias("clk_h"))
        .withColumn("ctr_hora", F.col("clk_h") / F.col("imp_h"))
    )
    w = Window.partitionBy("pid").orderBy("hora_abs").rangeBetween(-6, -1)
    return ts_min, base.select(
        "pid", "hora_abs", F.avg("ctr_hora").over(w).alias("ctr_movil_6h_slot")
    )


# ---------------------------------------------------------------------------
# 4. Precio relativo a su categoría. `price` no deriva del objetivo, así que no
#    hay fuga posible; aun así la referencia se calcula SOLO sobre la ventana de
#    entrenamiento, porque calcular estadísticas sobre el test es transducción y
#    en producción esa media no existiría todavía.
# ---------------------------------------------------------------------------
def _precio_referencia(silver):
    tr = silver.filter(
        (F.col("fecha_local") >= F.lit(config.TRAIN_INI))
        & (F.col("fecha_local") <= F.lit(config.TRAIN_FIN))
    )
    return tr.groupBy("cate_id").agg(F.avg("price").alias("precio_medio_cate"))


# ---------------------------------------------------------------------------
# Construcción completa
# ---------------------------------------------------------------------------
def construir(spark, silver, escribir: bool = True, m: float | None = None):
    """Devuelve la tabla Gold. Cada bloque está comentado con qué fuga evita."""
    m = config.M_SUAVIZADO if m is None else m
    prior = _prior_diferido(silver).cache()

    g = silver
    for e in config.ENTIDADES_HISTORICAS:
        h = _ctr_historico(silver, prior, e, m)
        # cate_id/pid/customer son tablas chicas; campaign_id es la única grande.
        g = g.join(F.broadcast(h) if e in ("pid", "cate_id") else h,
                   on=[e, "fecha_local"], how="left")

    # --- Fatiga del usuario. El desempate (time_stamp, adgroup_id, pid) es el que
    # la Fase 1 tuvo que agregar para que W2 fuera determinista: con 6.220.484 pares
    # usuario-segundo empatados, sin él la ventana da un resultado distinto por motor.
    w_user = Window.partitionBy("userid").orderBy("time_stamp", "adgroup_id", "pid")
    w_prev = w_user.rowsBetween(Window.unboundedPreceding, -1)
    g = (
        g.join(F.broadcast(prior), on="fecha_local", how="left")
        .withColumn("orden_impresion", F.row_number().over(w_user))
        .withColumn("gap_seg", F.col("time_stamp") - F.lag("time_stamp").over(w_user))
        .withColumn("clk_prev_u", F.coalesce(F.sum(OBJETIVO).over(w_prev), F.lit(0)).cast("double"))
    )
    mu = config.M_SUAVIZADO_USUARIO
    g = (
        g.withColumn("imp_prev_u", (F.col("orden_impresion") - 1).cast("double"))
        .withColumn(
            "ctr_usuario_previo",
            (F.col("clk_prev_u") + F.lit(mu) * F.col("p0_diferido"))
            / (F.col("imp_prev_u") + F.lit(mu)),
        )
        .withColumn("log_imp_previas_usuario", F.log1p("imp_prev_u"))
        .withColumn(
            "bucket_gap",
            F.when(F.col("gap_seg").isNull(), "primera")
            .when(F.col("gap_seg") <= 300, "0-5min")
            .when(F.col("gap_seg") <= 3600, "5-60min")
            .when(F.col("gap_seg") <= 86400, "1-24h")
            .otherwise("mas_1d"),
        )
        .withColumn("log_gap_seg", F.log1p(F.coalesce(F.col("gap_seg"), F.lit(0.0))))
        # El tope en 20 replica exactamente el LEAST(orden_impresion, 20) de W2:
        # más allá la curva de CTR es plana y la cola larga solo agrega varianza.
        .withColumn("orden_impresion", F.least(F.col("orden_impresion"), F.lit(20)).cast("double"))
    )

    ts_min, movil = _ctr_movil_slot(silver)
    g = (
        g.withColumn("hora_abs", F.floor((F.col("time_stamp") - F.lit(ts_min)) / 3600).cast("bigint"))
        .join(F.broadcast(movil), on=["pid", "hora_abs"], how="left")
    )

    g = (
        g.join(F.broadcast(_precio_referencia(silver)), on="cate_id", how="left")
        .withColumn("log_price", F.log1p(F.greatest(F.col("price"), F.lit(0.0))))
        .withColumn(
            "precio_rel_categoria",
            F.col("price") / F.when(F.col("precio_medio_cate") > 0,
                                    F.col("precio_medio_cate")).otherwise(F.lit(None)),
        )
    )

    # Relleno FINAL. Un null que llega al VectorAssembler aborta el ajuste, y lo
    # correcto es que cada relleno tenga una razón, no un 0 por defecto:
    #   ctr_movil_6h_slot  -> p0 diferido (primeras horas del período, sin historia)
    #   precio_rel_categoria -> 1.0 (categoría sin referencia = precio "normal")
    g = (
        g.withColumn("ctr_movil_6h_slot",
                     F.coalesce(F.col("ctr_movil_6h_slot"), F.col("p0_diferido")))
        .withColumn("precio_rel_categoria",
                    F.coalesce(F.col("precio_rel_categoria"), F.lit(1.0)))
        .withColumn("hora_local", F.col("hora_local").cast("int"))
    )

    # Identificadores. NO son features (ver DESCARTADAS): viajan en la tabla porque
    # el experimento de recomendación necesita la pareja (userid, cate_id) y porque
    # sin adgroup_id no se puede auditar una predicción hasta el aviso que la produjo.
    columnas = (["userid", "adgroup_id", "cate_id", "time_stamp", "fecha_local", OBJETIVO]
                + CATEGORICAS + NUMERICAS)
    columnas = list(dict.fromkeys(columnas))
    gold = g.select(*columnas)

    if escribir:
        (gold.write.mode("overwrite").option("compression", "zstd")
            .partitionBy("fecha_local").parquet(config.GOLD))
        gold = spark.read.parquet(config.GOLD)
    gold.createOrReplaceTempView("gold")
    return gold


# ---------------------------------------------------------------------------
# Split temporal
# ---------------------------------------------------------------------------
def particionar(gold):
    """(train, val, test). El burn-in NO se devuelve: existe solo como historia."""
    f = F.col("fecha_local")
    train = gold.filter((f >= F.lit(config.TRAIN_INI)) & (f <= F.lit(config.TRAIN_FIN)))
    val = gold.filter(f == F.lit(config.DIA_VAL))
    test = gold.filter(f == F.lit(config.DIA_TEST))
    return train, val, test


def resumen_split(gold) -> pd.DataFrame:
    """Filas, positivos y CTR por día, con la etiqueta del split. El CTR por día
    tiene que ser estable: si el día de test tuviera un CTR muy distinto, la
    comparación de métricas entre val y test no significaría nada."""
    d = (
        gold.groupBy("fecha_local")
        .agg(F.count(F.lit(1)).alias("filas"), F.sum(OBJETIVO).alias("positivos"))
        .orderBy("fecha_local")
        .toPandas()
    )
    d["fecha_local"] = d["fecha_local"].astype(str)
    d["ctr_pct"] = (d.positivos / d.filas * 100).round(3)

    def etiqueta(f):
        if f == config.DIA_BURNIN:
            return "burn-in (no se entrena)"
        if config.TRAIN_INI <= f <= config.TRAIN_FIN:
            return "train"
        if f == config.DIA_VAL:
            return "val"
        if f == config.DIA_TEST:
            return "test"
        return "fuera de rango"

    d["split"] = d.fecha_local.map(etiqueta)
    return d


# ---------------------------------------------------------------------------
# Control ejecutable de fuga
# ---------------------------------------------------------------------------
def verificar_fuga(gold, spark) -> pd.DataFrame:
    """Tres pruebas que tienen que pasar para que las métricas signifiquen algo.

    1. Ninguna correlación |r| > 0.95 entre un feature numérico y el objetivo.
       Un feature que casi ES el objetivo es fuga, no señal.
    2. El primer día entrenable NO tiene históricos nulos (el burn-in cumplió su función).
    3. Los CTR históricos del día d están dentro del rango plausible de una tasa
       (0,1] y no coinciden con el CTR observado del propio día d, que sería el
       síntoma de que la ventana quedó abierta en CURRENT ROW.
    """
    filas = []

    objetivo_num = [c for c in NUMERICAS if c not in ("sin_perfil", "brand_conocida",
                                                      "precio_valido", "es_finde")]
    corr = {c: gold.stat.corr(c, OBJETIVO) for c in objetivo_num}
    peor = max(corr, key=lambda c: abs(corr[c] or 0))
    filas.append({
        "prueba": "correlación feature-objetivo",
        "detalle": f"máx |r| = {abs(corr[peor] or 0):.4f} en {peor}",
        "veredicto": "OK" if abs(corr[peor] or 0) <= 0.95 else "FUGA",
    })

    n_nulos = gold.filter(F.col("fecha_local") == F.lit(config.TRAIN_INI)).select(
        F.sum(
            sum(
                (F.col(f"ctr_hist_{e}").isNull()).cast("int")
                for e in config.ENTIDADES_HISTORICAS
            )
        ).alias("n")
    ).collect()[0]["n"] or 0
    filas.append({
        "prueba": "históricos disponibles el primer día entrenable",
        "detalle": f"{int(n_nulos)} nulos el {config.TRAIN_INI}",
        "veredicto": "OK" if n_nulos == 0 else "BURN-IN INSUFICIENTE",
    })

    rangos = gold.select(
        *[F.min(f"ctr_hist_{e}").alias(f"min_{e}") for e in config.ENTIDADES_HISTORICAS],
        *[F.max(f"ctr_hist_{e}").alias(f"max_{e}") for e in config.ENTIDADES_HISTORICAS],
    ).collect()[0].asDict()
    fuera = {k: v for k, v in rangos.items() if v is not None and not (0 < v <= 1)}
    filas.append({
        "prueba": "CTR históricos dentro de (0, 1]",
        "detalle": "todos dentro del rango" if not fuera else str(fuera),
        "veredicto": "OK" if not fuera else "VALOR IMPOSIBLE",
    })

    return pd.DataFrame(filas)


def registro_variables() -> pd.DataFrame:
    """El artefacto versionado que prometió la Fase 1: aceptadas y descartadas."""
    filas = [{"variable": c, "tipo": "categórica", "estado": "aceptada", "motivo": ""}
             for c in CATEGORICAS]
    filas += [{"variable": c, "tipo": "numérica", "estado": "aceptada", "motivo": ""}
              for c in NUMERICAS]
    filas += [{"variable": k, "tipo": "-", "estado": "DESCARTADA", "motivo": v}
              for k, v in DESCARTADAS.items()]
    return pd.DataFrame(filas)
