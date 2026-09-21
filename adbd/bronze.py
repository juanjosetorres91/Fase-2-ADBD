"""
BRONZE · CSV crudo -> Parquet + ZSTD particionado por fecha_local.

Es exactamente la capa que produjo la Fase 1 y no se reescribe: si el directorio
ya existe con el contrato cumplido, se reutiliza. La reproducibilidad de extremo
a extremo que pide la Fase 3 exige que esta función pueda reconstruirlo todo
desde los CSV, así que existe y se puede forzar con `rehacer=True`.

La corrección de zona horaria (UTC -> UTC+8) se aplica AQUÍ, una sola vez.
Sin ella todo feature horario del modelo queda desplazado 8 horas.
"""
from __future__ import annotations

import os

from pyspark.sql import functions as F

from . import config, contrato
from .utilidades import tam_mb


def construir(spark, csv: dict[str, str] | None = None, rehacer: bool = False) -> dict:
    """Escribe (o reutiliza) la capa Bronze. Devuelve métricas para el informe."""
    res: dict = {}
    listo = all(
        os.path.exists(p)
        for p in (config.BRONZE_IMPRESIONES, config.BRONZE_ADS, config.BRONZE_USR)
    )
    if listo and not rehacer and _contrato_ok(spark):
        print("Bronze ya existe y cumple el contrato · se reutiliza (Fase 1)")
        res["bronze_reutilizado"] = True
    else:
        csv = csv or contrato.resolver_csv()
        res["bronze_reutilizado"] = False
        import time

        t0 = time.perf_counter()
        raw = (
            contrato.leer_csv(spark, "raw_sample", csv)[1]
            .withColumnRenamed("user", "userid")
            .withColumn("ts_utc", F.to_timestamp(F.from_unixtime("time_stamp")))
            .withColumn("ts_local", F.col("ts_utc") + F.expr("INTERVAL 8 HOURS"))
            .withColumn("fecha_local", F.to_date("ts_local"))
            .withColumn("hora_local", F.hour("ts_local"))
            .withColumn(
                "franja",
                F.when(F.col("hora_local") < 6, "madrugada")
                .when(F.col("hora_local") < 12, "manana")
                .when(F.col("hora_local") < 19, "tarde")
                .otherwise("noche"),
            )
        )
        (raw.write.mode("overwrite").option("compression", "zstd")
            .partitionBy("fecha_local").parquet(config.BRONZE_IMPRESIONES))
        for n, destino in (("ad_feature", config.BRONZE_ADS), ("user_profile", config.BRONZE_USR)):
            (contrato.leer_csv(spark, n, csv)[1].write.mode("overwrite")
                .option("compression", "zstd").parquet(destino))
        res["t_conversion"] = round(time.perf_counter() - t0, 1)
        assert _contrato_ok(spark), "el Parquet escrito no cumple el contrato"

    mb = tam_mb(config.BRONZE_IMPRESIONES)
    particiones = [
        p for p in os.listdir(config.BRONZE_IMPRESIONES) if p.startswith("fecha_local=")
    ]
    res["mb_bronze"] = round(mb, 1)
    res["n_particiones"] = len(particiones)
    res["particiones"] = sorted(p.split("=")[1] for p in particiones)
    print(f"Bronze: {res['mb_bronze']:,} MB en {res['n_particiones']} particiones")
    return res


def _contrato_ok(spark) -> bool:
    """El Parquet de una corrida anterior con esquema incompleto sobrevive en disco.
    Se verifica en vez de suponerlo."""
    try:
        esperado = {
            config.BRONZE_IMPRESIONES: {"userid", "time_stamp", "adgroup_id", "pid", "clk",
                                        "ts_local", "fecha_local", "hora_local", "franja"},
            config.BRONZE_ADS: contrato.REQUERIDAS["ad_feature"],
            config.BRONZE_USR: contrato.REQUERIDAS["user_profile"],
        }
        for ruta, req in esperado.items():
            if not req.issubset(set(spark.read.parquet(ruta).columns)):
                return False
        return True
    except Exception:
        return False


def leer(spark):
    """Las tres tablas Bronze, con vistas SQL registradas."""
    imp = spark.read.parquet(config.BRONZE_IMPRESIONES)
    ads = spark.read.parquet(config.BRONZE_ADS)
    usr = spark.read.parquet(config.BRONZE_USR)
    for df, n in ((imp, "impresiones"), (ads, "ad_feature"), (usr, "user_profile")):
        df.createOrReplaceTempView(n)
    return imp, ads, usr
