"""
adbd · Proyecto Integrador de Análisis de Big Data · Magíster en Data Science UDD
Equipo: Juan José Torres · Claudio Ballerini · Cristian Vargas · Christian Vásquez

Fase 2 — pipeline batch + ML distribuido + MLflow.
Las capas siguen la arquitectura decidida y medida en la Fase 1:

    CSV crudo -> BRONZE (Parquet+ZSTD por fecha_local) -> SILVER -> GOLD -> modelos

Cada módulo tiene una sola responsabilidad y se puede correr aislado, que es lo
que hace posible el `scripts/correr_fase2.py` de punta a punta.
"""

__version__ = "2.0.0"

from . import (  # noqa: F401
    config,
    contrato,
    bronze,
    silver,
    gold,
    transformadores,
    features,
    modelos,
    evaluacion,
    seguimiento,
    utilidades,
)
