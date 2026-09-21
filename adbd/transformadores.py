"""
Transformers propios. Dos, y cada uno existe porque MLlib no trae el equivalente.

`CodificadorCentinela` es higiene: un null que llega a StringIndexer con
handleInvalid="error" aborta el ajuste, y con "skip" BORRA la fila en silencio —
que es justo el fallo de calidad que la Fase 1 persiguió en la línea de carga.
Aquí el null se convierte en una categoría explícita antes de indexar.

`CorrectorPrior` es la pieza analítica del módulo. Al submuestrear negativos, el
modelo aprende sobre una prevalencia inventada (35% en vez de 5,14%) y sus
probabilidades quedan sistemáticamente infladas. El orden de las predicciones no
cambia —por eso AUC-ROC sobrevive intacto—, pero cualquier decisión que compare la
probabilidad contra un umbral, y cualquier métrica de calibración (LogLoss, Brier),
quedan mal. La corrección es exacta y tiene una línea:

    odds_reales = odds_submuestreadas · r        con r = tasa de negativos conservada
    p           = p_s·r / (p_s·r + 1 − p_s)

Comprobación: r = 1 devuelve p = p_s; r → 0 devuelve p → 0. Ambas se prueban en
tests/test_humo.py, porque una fórmula de calibración mal puesta no falla, miente.
"""
from __future__ import annotations

from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.functions import vector_to_array
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCol, HasInputCols, HasOutputCol, HasOutputCols
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import functions as F


class CodificadorCentinela(
    Transformer, HasInputCols, HasOutputCols, DefaultParamsReadable, DefaultParamsWritable
):
    """Castea a string y reemplaza nulls por un centinela explícito."""

    centinela = Param(
        Params._dummy(), "centinela", "token para el valor ausente",
        typeConverter=TypeConverters.toString,
    )

    @keyword_only
    def __init__(self, inputCols=None, outputCols=None, centinela="DESCONOCIDO"):
        super().__init__()
        self._setDefault(centinela="DESCONOCIDO")
        self.setParams(**self._input_kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCols=None, centinela="DESCONOCIDO"):
        return self._set(**self._input_kwargs)

    def getCentinela(self):
        return self.getOrDefault(self.centinela)

    def _transform(self, df):
        entradas = self.getInputCols()
        salidas = self.getOutputCols() or [f"{c}_cat" for c in entradas]
        token = self.getCentinela()
        for ent, sal in zip(entradas, salidas):
            df = df.withColumn(
                sal, F.coalesce(F.col(ent).cast("string"), F.lit(token))
            )
        return df


class CorrectorPrior(
    Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable
):
    """Devuelve la probabilidad a la prevalencia real tras submuestrear negativos.

    inputCol  : columna `probability` (Vector) o una columna double con p(clk=1)
    outputCol : probabilidad recalibrada (double)
    """

    tasaNegativos = Param(
        Params._dummy(), "tasaNegativos",
        "fracción de negativos conservada en el entrenamiento (r en (0, 1])",
        typeConverter=TypeConverters.toFloat,
    )

    @keyword_only
    def __init__(self, inputCol="probability", outputCol="p_calibrada", tasaNegativos=1.0):
        super().__init__()
        self._setDefault(inputCol="probability", outputCol="p_calibrada", tasaNegativos=1.0)
        self.setParams(**self._input_kwargs)

    @keyword_only
    def setParams(self, inputCol="probability", outputCol="p_calibrada", tasaNegativos=1.0):
        return self._set(**self._input_kwargs)

    def getTasaNegativos(self):
        return self.getOrDefault(self.tasaNegativos)

    def _transform(self, df):
        ent, sal = self.getInputCol(), self.getOutputCol()
        r = float(self.getTasaNegativos())
        if not 0 < r <= 1:
            raise ValueError(f"tasaNegativos tiene que estar en (0, 1]; llegó {r}")
        tipo = dict(df.dtypes)[ent]
        # `probability` de MLlib es un Vector; una columna ya escalar entra tal cual.
        ps = (F.col(ent).cast("double") if tipo in ("double", "float")
              else vector_to_array(F.col(ent))[1])
        return df.withColumn(
            sal, (ps * F.lit(r)) / (ps * F.lit(r) + (F.lit(1.0) - ps))
        )


def recalibrar(p_submuestreada: float, tasa_negativos: float) -> float:
    """Misma fórmula, en Python puro, para poder probarla sin levantar Spark."""
    r = tasa_negativos
    return (p_submuestreada * r) / (p_submuestreada * r + (1.0 - p_submuestreada))
