"""
Contrato de lectura. Es el mismo de la Fase 1, sin cambios de fondo: el esquema
se declara, no se infiere, y la ausencia de una columna requerida DETIENE la
ejecución en vez de borrarla en silencio.

Dos cosas que este contrato ya se cobró en la Fase 1 y que siguen vigentes:
  1. El espejo publica 'new_user_class_level ' con un espacio final.
  2. El cast a double fabricó 246.330 nulos en ad_feature.brand, que NO existen
     en el origen. Por eso `brand` no entra como predictor en la Fase 2 (ver gold.py):
     modelar esa columna sería modelar un artefacto de la línea de carga.
"""
from __future__ import annotations

import glob
import os
import re

from pyspark.sql import functions as F

from . import config

ESQUEMA = {
    "raw_sample": {
        "user": "bigint", "time_stamp": "bigint", "adgroup_id": "bigint",
        "pid": "string", "nonclk": "int", "clk": "int",
    },
    "ad_feature": {
        "adgroup_id": "bigint", "cate_id": "bigint", "campaign_id": "bigint",
        "customer": "bigint", "brand": "double", "price": "double",
    },
    "user_profile": {
        "userid": "bigint", "cms_segid": "int", "cms_group_id": "int",
        "final_gender_code": "int", "age_level": "int", "pvalue_level": "double",
        "shopping_level": "int", "occupation": "int", "new_user_class_level": "double",
    },
}

REQUERIDAS = {
    "raw_sample": {"user", "time_stamp", "adgroup_id", "pid", "clk"},
    "ad_feature": {"adgroup_id", "cate_id", "campaign_id", "customer", "price"},
    "user_profile": {"userid", "cms_segid", "cms_group_id", "final_gender_code",
                     "age_level", "pvalue_level", "shopping_level", "occupation",
                     "new_user_class_level"},
}


def _clave(c: str) -> str:
    """Normaliza encabezados para comparar: sin BOM, sin espacios, en minúsculas."""
    return re.sub(r"\s+", "", c.replace("﻿", "")).lower()


def localizar(nombre: str, ruta_csv: str | None = None) -> str | None:
    ruta_csv = ruta_csv or config.RUTA_CSV
    hits = glob.glob(os.path.join(ruta_csv, "**", config.ARCHIVOS[nombre]), recursive=True)
    return hits[0] if hits else None


def resolver_csv(ruta_csv: str | None = None) -> dict[str, str]:
    """Devuelve {tabla: ruta}. Si falta alguno, baja el espejo de Kaggle.
    La licencia reconocida es la de Tianchi; el espejo es solo un medio de acceso."""
    ruta_csv = ruta_csv or config.RUTA_CSV
    if not all(localizar(n, ruta_csv) for n in config.ARCHIVOS):
        # El fallo que más confunde aquí no es "falta el dataset" sino
        # "ModuleNotFoundError: kagglehub" a 20 líneas de profundidad, cuando en
        # realidad los CSV estaban en disco y quien llamó apuntó a la carpeta
        # equivocada. El mensaje tiene que decir las dos cosas.
        try:
            import kagglehub
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"no hay CSV en {os.path.abspath(ruta_csv)} y kagglehub no esta instalado.\n"
                f"  · directorio de trabajo actual: {os.getcwd()}\n"
                "  · si los datos ya estan en otra carpeta: ADBD_RUTA_CSV=<ruta> "
                "(o config.RUTA_CSV = <ruta>)\n"
                "  · para bajarlos del espejo de Kaggle: pip install kagglehub"
            ) from e

        ruta_csv = kagglehub.dataset_download(config.ESPEJO_KAGGLE)
        print("descargado en", ruta_csv)
    csv = {n: localizar(n, ruta_csv) for n in config.ARCHIVOS}
    faltan = [n for n, p in csv.items() if p is None]
    if faltan:
        raise FileNotFoundError(f"no se encontraron los CSV: {faltan} bajo {ruta_csv}")
    return csv


def leer_csv(spark, nombre: str, csv: dict[str, str], verboso: bool = True):
    """Lee un CSV aplicando el contrato. Devuelve (crudo, casteado, mapa_de_nombres)."""
    cols = ESQUEMA[nombre]
    crudo = spark.read.option("header", True).option("inferSchema", False).csv(csv[nombre])
    reales = {_clave(c): c for c in crudo.columns}
    resuelto, ausentes, saneadas = {}, [], []
    for c, t in cols.items():
        orig = reales.get(_clave(c))
        if orig is None:
            ausentes.append(c)
            continue
        resuelto[c] = (orig, t)
        if orig != c:
            saneadas.append(f"{orig!r} -> {c}")
    faltan = sorted(set(ausentes) & REQUERIDAS[nombre])
    if faltan:
        raise ValueError(
            f"{nombre}: faltan columnas REQUERIDAS: {faltan}. Encabezado real: {crudo.columns}"
        )
    if verboso and ausentes:
        print(f"  aviso · {nombre}: declaradas y ausentes -> {sorted(set(ausentes) - set(faltan))}")
    if verboso and saneadas:
        print(f"  encabezado saneado · {nombre}: {saneadas}")
    df = crudo.select([F.col(f"`{o}`").cast(t).alias(c) for c, (o, t) in resuelto.items()])
    return crudo, df, {c: o for c, (o, _) in resuelto.items()}


def auditar_cast(spark, csv: dict[str, str]):
    """Nulos ANTES vs DESPUÉS del cast. La diferencia la creó la línea de carga,
    no el origen, y hay que saberlo antes de que alguien use esa columna como feature."""
    import pandas as pd

    filas = []
    for nombre in ESQUEMA:
        crudo, cast, mapa = leer_csv(spark, nombre, csv, verboso=False)
        n_antes = crudo.select(
            [F.sum(F.col(f"`{o}`").isNull().cast("long")).alias(c) for c, o in mapa.items()]
        ).collect()[0].asDict()
        n_desp = cast.select(
            [F.sum(F.col(c).isNull().cast("long")).alias(c) for c in mapa]
        ).collect()[0].asDict()
        for c in mapa:
            filas.append((nombre, c, n_antes[c], n_desp[c], n_desp[c] - n_antes[c]))
    return pd.DataFrame(
        filas, columns=["tabla", "columna", "nulos_antes", "nulos_despues", "creados_por_el_cast"]
    )
