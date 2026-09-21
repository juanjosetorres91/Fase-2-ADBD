"""
Modelamiento distribuido. Cuatro experimentos supervisados y uno de recomendación.

El desbalance (CTR 5,19% en la ventana de entrenamiento, ~1:18) se ataca por
SUBMUESTREO DE NEGATIVOS, no por sobremuestreo ni por `weightCol` a secas, y la
razón es de costo medido: el train son 16.744.897 filas y las 12 combinaciones que
suman las grillas de los cuatro experimentos, con `CrossValidator` de 3 pliegues,
son 36 ajustes completos más los cuatro reajustes finales. El submuestreo baja el
entrenamiento a 2.458.575 filas conservando TODOS los positivos.

Lo que el submuestreo cuesta se paga y se declara:
  · la probabilidad queda inflada  -> se recalibra (transformadores.CorrectorPrior);
  · la evaluación se hace SIEMPRE sobre el conjunto completo sin submuestrear,
    porque AUC-PR depende de la prevalencia y medirlo sobre la muestra daría un
    número bonito que no existe en producción.

El experimento `lr_pesos_completo` es la contraprueba, y en la corrida sobre el
dataset real dio el veredicto más limpio posible: **AUC-PR 0,093432 con weightCol
sobre las 16,7M de filas contra 0,093534 con submuestreo sobre 2,46M** — una
diferencia de −0,0001, dentro del ruido— y **197,1 s contra 43,0 s, 4,6x el
tiempo**. El atajo no cuesta precisión; compra un factor 4,6 de cómputo.
"""
from __future__ import annotations

import time

from pyspark.ml.classification import GBTClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import functions as F

from . import config, features as feat, gold as gold_mod


# ---------------------------------------------------------------------------
# Desbalance
# ---------------------------------------------------------------------------
def submuestrear_negativos(df, tasa: float | None = None, semilla: int | None = None):
    """Conserva TODOS los positivos y una fracción `tasa` de los negativos.

    Devuelve (df_submuestreado, info). `info["tasa_efectiva"]` es la fracción de
    negativos realmente conservada, que no es exactamente `tasa` porque
    `sample` es de Bernoulli por fila. La recalibración usa la efectiva, no la
    pedida: con la nominal la corrección queda sesgada en el tercer decimal.
    """
    tasa = config.TASA_NEGATIVOS if tasa is None else tasa
    semilla = config.SEMILLA if semilla is None else semilla
    pos = df.filter(F.col(gold_mod.OBJETIVO) == 1)
    neg = df.filter(F.col(gold_mod.OBJETIVO) == 0)
    n_neg = neg.count()
    neg_s = neg.sample(withReplacement=False, fraction=tasa, seed=semilla)
    n_neg_s = neg_s.count()
    n_pos = pos.count()
    muestra = pos.unionByName(neg_s)
    info = {
        "tasa_nominal": tasa,
        "tasa_efectiva": n_neg_s / n_neg if n_neg else 1.0,
        "positivos": n_pos,
        "negativos_originales": n_neg,
        "negativos_conservados": n_neg_s,
        "filas_entrenamiento": n_pos + n_neg_s,
        "prevalencia_muestra": n_pos / (n_pos + n_neg_s),
        "prevalencia_real": n_pos / (n_pos + n_neg),
    }
    return muestra, info


# ---------------------------------------------------------------------------
# Definición de los experimentos
# ---------------------------------------------------------------------------
def catalogo_experimentos(numericas_reducidas=None):
    """Cada experimento responde UNA pregunta. Un barrido de modelos sin pregunta
    produce una tabla que no se puede defender ante el panel."""
    lr = LogisticRegression(labelCol=gold_mod.OBJETIVO, featuresCol="features",
                            maxIter=50, family="binomial")
    gbt = GBTClassifier(labelCol=gold_mod.OBJETIVO, featuresCol="features",
                        maxIter=40, maxDepth=5, stepSize=0.1, seed=config.SEMILLA)
    rf = RandomForestClassifier(labelCol=gold_mod.OBJETIVO, featuresCol="features",
                                numTrees=60, maxDepth=8, seed=config.SEMILLA,
                                subsamplingRate=0.7)

    return {
        "lr_contexto": {
            "pregunta": "¿Cuánto se predice SIN historia, solo con perfil y contexto?",
            "estimador": lr,
            "escalar": True,
            "numericas": numericas_reducidas,
            "grid": (ParamGridBuilder()
                     .addGrid(lr.regParam, [0.0, 0.01])
                     .addGrid(lr.elasticNetParam, [0.0])
                     .build()),
        },
        "lr_completo": {
            "pregunta": "¿Cuánto agregan los CTR históricos diferidos y la fatiga?",
            "estimador": lr,
            "escalar": True,
            "numericas": None,
            "grid": (ParamGridBuilder()
                     .addGrid(lr.regParam, [0.0, 0.001, 0.01])
                     .addGrid(lr.elasticNetParam, [0.0, 0.5])
                     .build()),
        },
        "rf_completo": {
            "pregunta": "¿Gana un ensemble de árboles sin interacciones explícitas?",
            "estimador": rf,
            "escalar": False,
            "numericas": None,
            "grid": (ParamGridBuilder()
                     .addGrid(rf.maxDepth, [6, 10])
                     .addGrid(rf.numTrees, [60])
                     .build()),
        },
        "gbt_completo": {
            "pregunta": "¿Compensa el boosting su costo de cómputo en AUC-PR?",
            "estimador": gbt,
            "escalar": False,
            "numericas": None,
            "grid": (ParamGridBuilder()
                     .addGrid(gbt.maxDepth, [4, 6])
                     .addGrid(gbt.maxIter, [40])
                     .build()),
        },
    }


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------
def ajustar_con_cv(train_ds, spec: dict, folds: int = 3,
                   fraccion_busqueda: float | None = None, metrica: str = "areaUnderPR"):
    """CrossValidator + ParamGridBuilder sobre la ventana de entrenamiento.

    Presupuesto de cómputo, declarado: la BÚSQUEDA corre sobre una submuestra
    aleatoria de `fraccion_busqueda` del train ya submuestreado; el modelo GANADOR
    se reajusta sobre el train submuestreado completo. Buscar sobre el 100% sería
    `folds x |grid|` ajustes completos y no cabe en la sesión.

    Sobre el k-fold aleatorio: parte la ventana de entrenamiento sin respetar el
    orden temporal, lo que normalmente sería una fuga. Aquí NO lo es, porque la
    protección temporal vive en los FEATURES (gold.py los construye con retardo),
    no en el corte: una fila del 10-may no contiene nada del 11-may aunque caiga
    en el mismo pliegue. La afirmación no se deja en palabras: `brecha_optimismo`
    compara la métrica de CV contra la del día de validación retenido, y esa
    diferencia es lo que el informe reporta.
    """
    fraccion_busqueda = config.FRACCION_BUSQUEDA if fraccion_busqueda is None else fraccion_busqueda
    pipe = feat.pipeline(spec["estimador"], numericas=spec.get("numericas"),
                         escalar=spec.get("escalar", False))
    ev = BinaryClassificationEvaluator(labelCol=gold_mod.OBJETIVO,
                                       rawPredictionCol="rawPrediction",
                                       metricName=metrica)
    cv = CrossValidator(estimator=pipe, estimatorParamMaps=spec["grid"], evaluator=ev,
                        numFolds=folds, parallelism=1, seed=config.SEMILLA,
                        collectSubModels=False)

    busqueda = (train_ds.sample(False, fraccion_busqueda, seed=config.SEMILLA)
                if fraccion_busqueda < 1 else train_ds)
    t0 = time.perf_counter()
    cvm = cv.fit(busqueda)
    t_busqueda = round(time.perf_counter() - t0, 1)

    mejor_idx = int(max(range(len(cvm.avgMetrics)), key=lambda i: cvm.avgMetrics[i]))
    mejores = {p.name: v for p, v in spec["grid"][mejor_idx].items()}

    t0 = time.perf_counter()
    modelo = pipe.copy(spec["grid"][mejor_idx]).fit(train_ds)
    t_ajuste = round(time.perf_counter() - t0, 1)

    info = {
        "n_combinaciones": len(spec["grid"]),
        "folds": folds,
        "fraccion_busqueda": fraccion_busqueda,
        "filas_busqueda": busqueda.count(),
        "metrica_cv": metrica,
        "cv_mejor": float(cvm.avgMetrics[mejor_idx]),
        "cv_todas": [float(m) for m in cvm.avgMetrics],
        "mejores_parametros": {k: (float(v) if isinstance(v, (int, float)) else str(v))
                               for k, v in mejores.items()},
        "t_busqueda_s": t_busqueda,
        "t_ajuste_final_s": t_ajuste,
        "ajustes_realizados": folds * len(spec["grid"]) + 1,
    }
    return modelo, info


# ---------------------------------------------------------------------------
# Línea base honesta
# ---------------------------------------------------------------------------
def agregar_pesos(df, col: str = "peso"):
    """Pesos balanceados para el experimento de contraste: el mismo modelo sobre
    el train COMPLETO, sin submuestrear, con `weightCol`. Es la contraprueba de
    que el submuestreo es un atajo de costo y no un truco para mejorar la métrica."""
    n = df.count()
    n_pos = df.filter(F.col(gold_mod.OBJETIVO) == 1).count()
    w_pos = (n - n_pos) / n_pos if n_pos else 1.0
    info = {"filas": n, "positivos": n_pos, "peso_positivo": round(w_pos, 3),
            "peso_negativo": 1.0}
    return df.withColumn(
        col, F.when(F.col(gold_mod.OBJETIVO) == 1, F.lit(float(w_pos))).otherwise(F.lit(1.0))
    ), info


def curva_aprendizaje(train_ds, spec, fracciones=(0.1, 0.25, 0.5, 1.0), evaluar=None,
                      corrector=None, metrica_fn=None):
    """¿Más datos o más modelo? Ajusta el mismo pipeline sobre fracciones
    crecientes del entrenamiento y registra AUC-PR y tiempo.

    Es la respuesta cuantitativa a la pregunta que la Fase 1 dejó abierta: si la
    curva ya está plana al 100% del dataset actual, incorporar `behavior_log`
    (26x el volumen) compra tiempo de cómputo y no precisión, y esa es una
    decisión de escalamiento defendible con un número en vez de una intuición.

    Resultado sobre el dataset real: de 245.848 a 2.458.575 filas —10x— el AUC-PR
    pasa de 0,091259 a 0,091470, **+0,23%**. La curva está plana, y la conclusión
    sobre `behavior_log` se sostiene en ese número. Ojo con lo que NO dice: está
    plana *para este espacio de features y este modelo*. Más volumen serviría si
    trajera features nuevas (el comportamiento de navegación que `behavior_log`
    contiene), no más filas de las mismas.
    """
    import pandas as pd

    pipe = feat.pipeline(spec["estimador"], numericas=spec.get("numericas"),
                         escalar=spec.get("escalar", False))
    filas = []
    for f in fracciones:
        sub = train_ds.sample(False, f, seed=config.SEMILLA) if f < 1 else train_ds
        n = sub.count()
        t0 = time.perf_counter()
        modelo = pipe.fit(sub)
        t = round(time.perf_counter() - t0, 1)
        pred = modelo.transform(evaluar)
        if corrector is not None:
            pred = corrector.transform(pred)
        m = metrica_fn(pred)
        filas.append({"fraccion": f, "filas": n, "t_ajuste_s": t,
                      "auc_pr": m["auc_pr"], "auc_roc": m["auc_roc"],
                      "filas_por_segundo": round(n / t) if t else None})
        print(f"  {f:>5.0%} · {n:>10,} filas · {t:>7.1f} s · AUC-PR {m['auc_pr']:.5f}")
    return pd.DataFrame(filas)


def linea_base_prior(train, evaluar):
    """Predice el CTR histórico constante para todas las impresiones.

    No es relleno: fija el piso real. Con la prevalencia de 5,03% del día de test,
    un clasificador que acierta el 94,97% de las veces diciendo siempre "no click"
    tiene accuracy excelente y AUC-PR igual a la prevalencia. Sin esta línea base,
    el AUC-PR de 0,0935 del mejor modelo parecería malo; contra el piso de 0,0503
    es **1,86x**, y eso es lo que se defiende.
    """
    p0 = train.agg(F.avg(gold_mod.OBJETIVO)).collect()[0][0]
    return p0, evaluar.withColumn("p_calibrada", F.lit(float(p0)))


# ---------------------------------------------------------------------------
# ALS · feedback implícito usuario x categoría
# ---------------------------------------------------------------------------
def matriz_implicita(df, col_item: str = "cate_id"):
    """clicks por (usuario, categoría) en la ventana dada. Solo clicks: una
    impresión sin click no es una señal negativa, es ausencia de evidencia —
    exactamente el supuesto que `implicitPrefs=True` modela."""
    return (df.filter(F.col(gold_mod.OBJETIVO) == 1)
              .groupBy("userid", col_item)
              .agg(F.count(F.lit(1)).cast("double").alias("clicks")))


def verificar_ids_als(*matrices) -> dict:
    """ALS exige enteros de 32 bits. `userid` y `cate_id` son bigint; que quepan
    es plausible pero no se supone: se mide el máximo y se aborta si no cabe.
    Si algún día no cupiera, la salida es indexar, no truncar en silencio."""
    LIMITE = 2_147_483_647
    maximos = {}
    for m in matrices:
        r = m.agg(F.max("userid").alias("u"), F.max("cate_id").alias("i")).collect()[0]
        maximos["userid"] = max(maximos.get("userid", 0), int(r["u"] or 0))
        maximos["cate_id"] = max(maximos.get("cate_id", 0), int(r["i"] or 0))
    fuera = {k: v for k, v in maximos.items() if v > LIMITE}
    assert not fuera, f"no caben en int32 y habría que indexar: {fuera}"
    return maximos


def entrenar_als(train_mat, rank=None, reg=None, alpha=None, iters=None):
    train_mat = train_mat.withColumn("userid", F.col("userid").cast("int")) \
                         .withColumn("cate_id", F.col("cate_id").cast("int"))
    als = ALS(
        userCol="userid", itemCol="cate_id", ratingCol="clicks",
        rank=rank or config.ALS_RANK,
        regParam=reg or config.ALS_REG,
        alpha=alpha or config.ALS_ALPHA,
        maxIter=iters or config.ALS_ITER,
        implicitPrefs=True, coldStartStrategy="drop",
        nonnegative=True, seed=config.SEMILLA,
    )
    t0 = time.perf_counter()
    modelo = als.fit(train_mat)
    return modelo, round(time.perf_counter() - t0, 1)


def recomendaciones_populares(train_mat, k: int = None):
    """Línea base de recomendación: las k categorías más clickeadas, iguales para
    todos. Sin ella, un MAP@10 de 0,175 no se puede leer.

    Y en esta corrida fue la que dio el veredicto: la popularidad **gana**
    (MAP@10 0,2194 contra 0,1755 de ALS, NDCG 0,2753 contra 0,2150) y además
    cubre al 100% de los usuarios del test contra el 55,05% de ALS. Es un
    resultado negativo legítimo y se reporta como tal: a nivel de `cate_id` la
    granularidad es demasiado gruesa para que la personalización pague."""
    k = k or config.TOP_K
    top = [r["cate_id"] for r in
           train_mat.groupBy("cate_id").agg(F.sum("clicks").alias("c"))
                    .orderBy(F.col("c").desc()).limit(k).collect()]
    return top
