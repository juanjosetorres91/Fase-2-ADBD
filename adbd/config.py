"""
Constantes del proyecto. Todo lo que el informe cita como "supuesto declarado" vive aquí,
en un solo lugar, para que no haya dos verdades sobre el mismo número.

Regla heredada de la Fase 1: ninguna afirmación se escribe sin una celda que la imprima.
Su corolario para la Fase 2: ninguna constante se escribe dos veces.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
RUTA_CSV = os.environ.get("ADBD_RUTA_CSV", "./datos_csv")
RUTA_PARQUET = os.environ.get("ADBD_RUTA_PARQUET", "./datos_parquet")
RUTA_MLRUNS = os.environ.get("ADBD_RUTA_MLRUNS", "./mlruns")
RUTA_ARTEFACTOS = os.environ.get("ADBD_RUTA_ARTEFACTOS", "./artefactos_fase2")

ESPEJO_KAGGLE = "pavansanagapati/ad-displayclick-data-on-taobaocom"
ARCHIVOS = {
    "raw_sample": "raw_sample.csv",
    "ad_feature": "ad_feature.csv",
    "user_profile": "user_profile.csv",
}

# Capas. La frontera es Parquet, como quedó decidido en la Fase 1 (sección 4.2).
BRONZE_IMPRESIONES = f"{RUTA_PARQUET}/impresiones"
BRONZE_ADS = f"{RUTA_PARQUET}/ad_feature"
BRONZE_USR = f"{RUTA_PARQUET}/user_profile"
SILVER = f"{RUTA_PARQUET}/silver_impresiones"
GOLD = f"{RUTA_PARQUET}/gold_ads_v1"

# ---------------------------------------------------------------------------
# Split temporal. NO es aleatorio y la razón es la única que importa:
# en producción el modelo predice el futuro, así que la evaluación tiene que
# hacer lo mismo. Un split aleatorio sobre datos con historia de usuario mezcla
# impresiones del mismo usuario entre train y test y regala señal.
#
# El dataset son 8 días LOCALES completos (2017-05-06 .. 2017-05-13, UTC+8).
#   BURN-IN  06-may : NO se entrena con él. Existe solo para que el primer día
#                     entrenable tenga historia y los features diferidos no sean nulos.
#   TRAIN    07..11-may (5 días)
#   VAL      12-may       selección de modelo e hiperparámetros
#   TEST     13-may       se toca UNA vez, al final
# ---------------------------------------------------------------------------
DIA_BURNIN = "2017-05-06"
TRAIN_INI, TRAIN_FIN = "2017-05-07", "2017-05-11"   # ambos inclusive
DIA_VAL = "2017-05-12"
DIA_TEST = "2017-05-13"

# ---------------------------------------------------------------------------
# Calidad de datos: los umbrales de la Fase 1, sin cambiarlos a conveniencia.
#   <= 1%  -> se puede descartar la fila
#   <= 5%  -> se imputa
#   >  5%  -> NO se filtra ni se imputa a ciegas: categoría explícita + flag,
#             y se MIDE si la ausencia es informativa antes de decidir.
# ---------------------------------------------------------------------------
UMBRAL_DESCARTE = 0.01
UMBRAL_IMPUTACION = 0.05

# ---------------------------------------------------------------------------
# Codificación de objetivo diferida (target encoding con retardo de un día).
# m es el peso del prior en el suavizado de m-estimación:
#     ctr_hist = (clicks_previos + m * p0) / (impresiones_previas + m)
# Con m = 200 una entidad necesita ~200 impresiones históricas para que su propia
# tasa pese la mitad. Es un supuesto declarado y la sección de sensibilidad lo mueve.
# ---------------------------------------------------------------------------
M_SUAVIZADO = 200.0
M_SUAVIZADO_USUARIO = 20.0     # el usuario tiene ~25 impresiones en total: m alto lo anularía

# adgroup_id queda FUERA de esta lista a propósito: 846.811 avisos x 8 días son 6,8M
# de filas de historia que hay que unir contra 26,6M de impresiones, y el nivel
# campaign_id lo contiene jerárquicamente (un adgroup pertenece a una campaña).
# Es una decisión de costo, declarada, no un olvido.
ENTIDADES_HISTORICAS = ["cate_id", "campaign_id", "customer", "pid"]

# ---------------------------------------------------------------------------
# Desbalance. CTR global 5,14% (~1:18). Se submuestrean NEGATIVOS y se
# RECALIBRA la probabilidad; los positivos nunca se tocan.
# ---------------------------------------------------------------------------
TASA_NEGATIVOS = 0.10          # se conserva 1 de cada 10 negativos en entrenamiento
FRACCION_BUSQUEDA = 0.20       # submuestra adicional SOLO para la búsqueda de hiperparámetros
SEMILLA = 42

# ---------------------------------------------------------------------------
# ALS (feedback implícito): usuario x categoría, clicks como confianza.
# ---------------------------------------------------------------------------
ALS_RANK = 32
ALS_REG = 0.05
ALS_ALPHA = 20.0
ALS_ITER = 10
TOP_K = 10

# ---------------------------------------------------------------------------
# FinOps. Misma tarifa que la Fase 1 para que las dos fases sean comparables.
# Es una REFERENCIA de dimensionamiento, no un gasto incurrido: el proyecto
# corre en Colab gratuito y el costo monetario efectivo es US$ 0.
# ---------------------------------------------------------------------------
TARIFA_USD_HORA = 0.27         # e2-standard-8 (8 vCPU / 32 GB), on-demand, ago-2026
FACTOR_BEHAVIOR = 704 / 26.6   # si entra behavior_log, el volumen se multiplica por esto

# Configuración de Spark. shuffle.partitions = 8 es la decisión de la Fase 1 para
# 1,1 GB en 2 núcleos; el default de 200 produce particiones de pocos MB.
SPARK_CONF = {
    "spark.sql.shuffle.partitions": "8",
    # 8g es lo que da Colab. ADBD_DRIVER_MEM permite bajarlo para la prueba de humo
    # o subirlo en Databricks sin tocar el código.
    "spark.driver.memory": os.environ.get("ADBD_DRIVER_MEM", "8g"),
    "spark.sql.session.timeZone": "UTC",
    "spark.sql.adaptive.enabled": "true",
    # Arrow acelera toPandas(). Se puede apagar con ADBD_ARROW=false: la combinación
    # Spark 3.5 + JDK 21 no la soporta (el Arrow que empaqueta Spark 3.5 es anterior),
    # y la prueba de humo de este repositorio corre en esa JVM.
    "spark.sql.execution.arrow.pyspark.enabled": os.environ.get("ADBD_ARROW", "true"),
}

# Arrow toca memoria fuera del heap por reflexión. Desde Java 17 el módulo
# java.nio está cerrado y `toPandas()` muere con
# "sun.misc.Unsafe or java.nio.DirectByteBuffer.<init>(long,int) not available".
# Colab trae Java 11 y no lo necesita; la JVM tiene que recibir estas banderas
# ANTES de arrancar, así que van por PYSPARK_SUBMIT_ARGS y no por .config()
# (ponerlas ahí en modo local se ignora en silencio, que es peor que fallar).
OPCIONES_JVM = (
    "--add-opens=java.base/java.nio=ALL-UNNAMED "
    "--add-opens=java.base/java.lang=ALL-UNNAMED "
    "--add-opens=java.base/java.util=ALL-UNNAMED "
    "-Dio.netty.tryReflectionSetAccessible=true"
)

NOMBRE_EXPERIMENTO = "adbd_fase2_ctr"
