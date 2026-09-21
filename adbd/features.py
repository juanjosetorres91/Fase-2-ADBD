"""
Ensamblado del Pipeline de MLlib: Transformers y Estimators, en ese orden.

    CodificadorCentinela  (Transformer propio)   null -> categoría explícita
    StringIndexer         (Estimator)            categoría -> índice, por columna
    OneHotEncoder         (Estimator)            índice -> vector disperso
    VectorAssembler       (Transformer)          todo -> un solo vector
    StandardScaler        (Estimator)            centrado/escala, solo si el modelo lo necesita

Dos decisiones que el informe defiende:

1. **Nada de alta cardinalidad entra codificado.** adgroup_id (846.811), campaign_id
   (423k) y userid (1,06M) NO se indexan: un OneHotEncoder sobre ellos produce un
   vector de millones de columnas y un StringIndexer sobre 1,06M de niveles es un
   `collect` del vocabulario al driver. Entran por sus agregados históricos
   diferidos (gold.py), que es justo la representación que sí generaliza a una
   entidad nueva.

2. **`handleInvalid="keep"` en ambos.** Una categoría que aparece en test y no en
   train existe de verdad (campañas nuevas todos los días) y tiene que recibir un
   índice, no tumbar la predicción ni desaparecer de la evaluación.

El escalado es condicional: la regresión logística con regularización lo necesita,
los árboles no lo usan y pagarlo sería tiempo regalado.
"""
from __future__ import annotations

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler

from . import gold as gold_mod
from .transformadores import CodificadorCentinela


def etapas_features(categoricas=None, numericas=None, escalar: bool = False,
                    salida: str = "features"):
    """Las etapas de preparación, sin el estimador final."""
    categoricas = categoricas or gold_mod.CATEGORICAS
    numericas = numericas or gold_mod.NUMERICAS

    cent = [f"{c}_cat" for c in categoricas]
    idx = [f"{c}_idx" for c in categoricas]
    ohe = [f"{c}_ohe" for c in categoricas]

    etapas = [
        CodificadorCentinela(inputCols=categoricas, outputCols=cent),
        StringIndexer(inputCols=cent, outputCols=idx, handleInvalid="keep",
                      stringOrderType="frequencyDesc"),
        OneHotEncoder(inputCols=idx, outputCols=ohe, handleInvalid="keep", dropLast=True),
    ]
    destino = "features_sin_escalar" if escalar else salida
    etapas.append(
        VectorAssembler(inputCols=ohe + list(numericas), outputCol=destino,
                        handleInvalid="error")
    )
    if escalar:
        etapas.append(
            StandardScaler(inputCol=destino, outputCol=salida,
                           withMean=False, withStd=True)
        )
    return etapas


def pipeline(estimador, categoricas=None, numericas=None, escalar: bool = False):
    return Pipeline(stages=etapas_features(categoricas, numericas, escalar) + [estimador])


def nombres_features(modelo_pipeline, categoricas=None, numericas=None) -> list[str]:
    """Nombres en el MISMO orden que el vector ensamblado, para poder leer los
    coeficientes y las importancias. Sin esto la interpretación es adivinanza.

    El ancho de cada bloque one-hot NO se deduce: se lee de `categorySizes` del
    OneHotEncoderModel ajustado, que es la única fuente que conoce el efecto
    combinado de dropLast y handleInvalid="keep". Deducirlo a mano desalinea los
    nombres por una posición y hace que el informe atribuya un coeficiente a la
    variable equivocada — un error que no se ve porque no falla.
    """
    categoricas = categoricas or gold_mod.CATEGORICAS
    numericas = numericas or gold_mod.NUMERICAS
    indexador = next(e for e in modelo_pipeline.stages
                     if type(e).__name__ == "StringIndexerModel")
    codificador = next(e for e in modelo_pipeline.stages
                       if type(e).__name__ == "OneHotEncoderModel")
    tamanos = list(codificador.categorySizes)
    dropea = bool(codificador.getDropLast())

    nombres = []
    for col, labels, n_cat in zip(categoricas, indexador.labelsArray, tamanos):
        etiquetas = [str(l) for l in labels] + ["__desconocido__"]
        ancho = n_cat - 1 if dropea else n_cat
        etiquetas = (etiquetas + [f"__nivel_{i}__" for i in range(ancho)])[:ancho]
        nombres += [f"{col}={l}" for l in etiquetas]
    return nombres + list(numericas)


def importancias(modelo_pipeline, categoricas=None, numericas=None, top: int = 25):
    """Coeficientes (modelos lineales) o importancias (árboles), con nombre.

    Sirve para dos cosas distintas y las dos importan en la defensa: explicar qué
    mueve la predicción, y detectar fuga que las métricas no delatan. Una variable
    que concentra casi toda la importancia suele ser una fuga, no un hallazgo.
    """
    import pandas as pd

    estimador = modelo_pipeline.stages[-1]
    nombres = nombres_features(modelo_pipeline, categoricas, numericas)
    if hasattr(estimador, "coefficients"):
        pesos = list(estimador.coefficients.toArray())
        clase = "coeficiente"
    elif hasattr(estimador, "featureImportances"):
        pesos = list(estimador.featureImportances.toArray())
        clase = "importancia"
    else:
        return pd.DataFrame()
    n = min(len(nombres), len(pesos))
    d = pd.DataFrame({"variable": nombres[:n], clase: pesos[:n]})
    d["magnitud"] = d[clase].abs()
    d["peso_relativo_pct"] = (d.magnitud / d.magnitud.sum() * 100).round(2)
    return d.sort_values("magnitud", ascending=False).head(top).reset_index(drop=True)
