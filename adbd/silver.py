"""
SILVER · impresiones ⋈ ad_feature ⋈ user_profile, con perfilamiento de calidad
y las decisiones de imputación TOMADAS CON UNA MEDICIÓN, no por costumbre.

La Fase 1 dejó dos columnas explícitamente pendientes para acá:
    pvalue_level          54,24% de nulos
    new_user_class_level  32,49% de nulos
La respuesta refleja no es imputar con la moda. Antes de decidir se mide si la
AUSENCIA es informativa: si el grupo sin dato tiene un CTR distinto del grupo con
dato, entonces "no sé" es información sobre el usuario y sustituirla por la moda
la destruye. `perfilar_calidad()` produce esa evidencia y `decidir_imputacion()`
aplica la regla de umbrales de la Fase 1 sobre ella.

Medido sobre el cruce real, los porcentajes NO son los de la Fase 1 y la
diferencia es del tipo que importa: `pvalue_level` tiene **54,80%** de ausencia en
el cruce (54,24% dentro de la tabla de perfiles) y `new_user_class_level`
**30,97%** (32,49% dentro de la tabla). La completitud de la tabla y la del cruce
son dos números distintos, y el que decide es el segundo, porque es el dato con el
que se entrena.
"""
from __future__ import annotations

import pandas as pd
from pyspark.sql import functions as F

from . import config
from .bronze import leer as leer_bronze

# Columnas de perfil que pueden faltar. -1 es el centinela de "desconocido":
# no es un valor válido de ninguna de ellas, así que no se confunde con un dato real.
CATEGORICAS_PERFIL = [
    "cms_segid", "cms_group_id", "final_gender_code", "age_level",
    "pvalue_level", "shopping_level", "occupation", "new_user_class_level",
]
CENTINELA = -1


def perfilar_calidad(spark) -> pd.DataFrame:
    """% de nulos por columna EN EL CRUCE (no dentro de la tabla de perfiles) y
    CTR del grupo ausente vs el presente. Es la tabla que decide la imputación.

    El hallazgo 1(b) de la Fase 1 fue exactamente esto: dentro de user_profile.csv
    final_gender_code tiene 0% de nulos, pero en el JOIN falta el 5,76%. La
    completitud relevante es la del cruce, porque es el dato con el que se entrena.
    """
    leer_bronze(spark)
    filas = []
    for c in CATEGORICAS_PERFIL:
        r = spark.sql(f"""
            SELECT
              COUNT(*)                                                   AS n,
              SUM(CASE WHEN u.{c} IS NULL THEN 1 ELSE 0 END)             AS n_nulos,
              SUM(CASE WHEN u.{c} IS NULL THEN i.clk ELSE 0 END)         AS clk_nulos,
              SUM(CASE WHEN u.{c} IS NOT NULL THEN i.clk ELSE 0 END)     AS clk_no_nulos
            FROM impresiones i LEFT JOIN user_profile u ON i.userid = u.userid
        """).collect()[0]
        n, nn = int(r["n"]), int(r["n_nulos"])
        ctr_aus = (r["clk_nulos"] / nn * 100) if nn else None
        ctr_pre = (r["clk_no_nulos"] / (n - nn) * 100) if n - nn else None
        filas.append({
            "columna": c,
            "pct_nulos_en_cruce": round(100 * nn / n, 2),
            "impresiones_sin_dato": nn,
            "ctr_sin_dato_pct": round(ctr_aus, 3) if ctr_aus is not None else None,
            "ctr_con_dato_pct": round(ctr_pre, 3) if ctr_pre is not None else None,
            "razon_ctr": round(ctr_aus / ctr_pre, 3) if ctr_aus and ctr_pre else None,
            "z": _z_dos_proporciones(int(r["clk_nulos"]), nn,
                                     int(r["clk_no_nulos"]), n - nn),
        })
    return pd.DataFrame(filas)


def _z_dos_proporciones(exitos_a: int, n_a: int, exitos_b: int, n_b: int):
    """z de la diferencia de dos proporciones, con varianza agrupada.

    Existe porque la primera versión de `decidir_imputacion` usaba un umbral fijo
    —"informativa si el CTR difiere más de 5%"— y sobre 26,6M de filas ese umbral
    está mal calibrado en las dos direcciones: declara *no informativa* una
    diferencia de 5,335% contra 5,132% que con 1,5M de observaciones tiene z = 11,
    y declararía informativa cualquier ruido medido sobre un grupo chico. El
    tamaño del efecto y la evidencia de que el efecto existe son dos preguntas
    distintas, y la regla de imputación necesita la segunda.
    """
    import math

    if not n_a or not n_b:
        return None
    pa, pb = exitos_a / n_a, exitos_b / n_b
    p = (exitos_a + exitos_b) / (n_a + n_b)
    se = math.sqrt(p * (1 - p) * (1 / n_a + 1 / n_b))
    return round((pa - pb) / se, 2) if se else None


def decidir_imputacion(perfil: pd.DataFrame, z_critico: float = 3.0) -> pd.DataFrame:
    """Aplica la regla de umbrales de la Fase 1 sobre la evidencia medida.

    <= 1%  descartar_fila          la ausencia es marginal
    <= 5%  imputar_moda            cuesta poco y no mueve la distribución
    >  5%  categoria_desconocido   + flag

    y en los tres casos, si la ausencia es INFORMATIVA —el CTR del grupo sin dato
    difiere del CTR del grupo con dato más allá del azar— gana
    `categoria_desconocido`, porque imputar destruiría esa señal.

    **Qué decide "informativa", y por qué no es el tamaño del efecto.** La primera
    versión de esta función usaba un umbral fijo sobre la razón de CTR (5%). Sobre
    26,6M de filas eso está mal calibrado: el grupo sin perfil tiene 5,335% de CTR
    contra 5,132% del grupo con perfil —una razón de 1,040, por debajo del umbral—
    y sin embargo la diferencia tiene **z = 11,0**. Con ese umbral, las ocho
    columnas salían marcadas *"ausencia no informativa"*, contradiciendo el
    hallazgo de la Fase 1 de que el segmento sin perfil clickea por encima del
    promedio. Lo que decide es si la diferencia **existe**, y eso es una prueba de
    dos proporciones; el tamaño del efecto se reporta aparte para que el lector
    juzgue si además importa.
    """
    def decidir(r):
        p = r.pct_nulos_en_cruce / 100
        if p == 0:
            return "sin_accion", "no hay ausencia en el cruce"
        z = r.get("z")
        informativa = z is not None and abs(z) > z_critico
        evidencia = (f"CTR ausente/presente = {r.razon_ctr} con z = {z}"
                     if z is not None else "sin evidencia de z")
        if p <= config.UMBRAL_DESCARTE:
            return ("categoria_desconocido" if informativa else "descartar_fila"), (
                f"{r.pct_nulos_en_cruce}% <= 1%"
                + (f"; {evidencia} -> la ausencia ES un dato" if informativa else "")
            )
        if p <= config.UMBRAL_IMPUTACION:
            return ("categoria_desconocido" if informativa else "imputar_moda"), (
                f"{r.pct_nulos_en_cruce}% <= 5%"
                + (f"; {evidencia} -> la ausencia ES un dato" if informativa else "")
            )
        return "categoria_desconocido", (
            f"{r.pct_nulos_en_cruce}% > 5%; " + evidencia
            + (" -> la ausencia ES un dato" if informativa
               else " -> no distinguible del azar, pero el volumen impide imputar")
        )

    d = perfil.copy()
    d[["decision", "motivo"]] = d.apply(lambda r: pd.Series(decidir(r)), axis=1)
    return d


def construir(spark, decisiones: pd.DataFrame | None = None, escribir: bool = True):
    """El cruce, con el flag sin_perfil y el centinela aplicado a las categóricas."""
    imp, ads, usr = leer_bronze(spark)

    silver = (
        imp.join(F.broadcast(ads), on="adgroup_id", how="inner")
           .join(usr, on="userid", how="left")
           .withColumn("sin_perfil", F.col("cms_segid").isNull().cast("int"))
    )
    # Centinela en todas las categóricas de perfil. Si la decisión de una columna
    # fue imputar_moda o descartar_fila, el notebook lo hace explícito antes de aquí;
    # el centinela es el default porque es la decisión que la Fase 1 ya tomó para
    # las dos columnas con más ausencia.
    for c in CATEGORICAS_PERFIL:
        silver = silver.withColumn(c, F.coalesce(F.col(c).cast("int"), F.lit(CENTINELA)))

    silver = (
        silver
        # brand quedó FUERA como predictor: el cast fabricó 246.330 nulos que no
        # existen en el origen (Fase 1, sección 2). Solo sobrevive como flag.
        .withColumn("brand_conocida", F.col("brand").isNotNull().cast("int"))
        .withColumn("precio_valido", (F.col("price") > 0).cast("int"))
        .withColumn("dia_semana", F.dayofweek("ts_local"))
        .withColumn("es_finde", F.col("dia_semana").isin([1, 7]).cast("int"))
        .drop("brand", "nonclk", "ts_utc")
    )

    if escribir:
        (silver.write.mode("overwrite").option("compression", "zstd")
            .partitionBy("fecha_local").parquet(config.SILVER))
        silver = spark.read.parquet(config.SILVER)
    silver.createOrReplaceTempView("silver")
    return silver


def validar_cardinalidad(spark, n_bronze: int) -> pd.DataFrame:
    """El control de la Fase 1, repetido: un JOIN mal especificado infla las filas
    y el CTR resultante SIGUE pareciendo razonable. delta != 0 detiene el pipeline."""
    leer_bronze(spark)
    pruebas = []
    for etiqueta, sql in [
        ("impresiones ⋈ ad_feature (INNER)",
         "SELECT COUNT(*) FROM impresiones i JOIN ad_feature a ON i.adgroup_id = a.adgroup_id"),
        ("+ LEFT JOIN user_profile",
         """SELECT COUNT(*) FROM impresiones i
            JOIN ad_feature a ON i.adgroup_id = a.adgroup_id
            LEFT JOIN user_profile u ON i.userid = u.userid"""),
    ]:
        n = spark.sql(sql).collect()[0][0]
        pruebas.append({"prueba": etiqueta, "filas": n, "delta": n - n_bronze,
                        "veredicto": "OK" if n == n_bronze else
                                     ("DUPLICA" if n > n_bronze else "PIERDE FILAS")})
    for tabla, clave in (("ad_feature", "adgroup_id"), ("user_profile", "userid")):
        tot, dis = spark.sql(f"SELECT COUNT(*), COUNT(DISTINCT {clave}) FROM {tabla}").collect()[0]
        pruebas.append({"prueba": f"{tabla}.{clave} único", "filas": tot, "delta": tot - dis,
                        "veredicto": "OK" if tot == dis else "CLAVE NO ÚNICA"})
    card = pd.DataFrame(pruebas)
    assert (card.veredicto == "OK").all(), f"los joins alteran la cardinalidad:\n{card}"
    return card
