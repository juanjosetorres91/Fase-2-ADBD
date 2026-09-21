"""
Prueba de humo. Corre el pipeline COMPLETO sobre datos sintéticos en ~1 minuto.

No prueba que el modelo sea bueno —con datos sintéticos no tendría sentido—, sino
que las piezas que pueden fallar en silencio no fallan:

  · el contrato de lectura sobrevive al encabezado sucio y al `brand` no numérico;
  · el split temporal deja el burn-in fuera del entrenamiento;
  · los features diferidos no tienen nulos el primer día entrenable;
  · la fórmula de recalibración es la que se cree que es (se prueba contra casos
    cerrados, no contra sí misma);
  · el submuestreo conserva TODOS los positivos.

    pytest tests/ -v          o          python tests/test_humo.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adbd.transformadores import recalibrar  # noqa: E402


# ---------------------------------------------------------------------------
# 1. La fórmula de recalibración, sin Spark de por medio.
# ---------------------------------------------------------------------------
def test_recalibrar_identidad():
    """Sin submuestreo (r = 1) la probabilidad no se toca."""
    for p in (0.01, 0.2, 0.5, 0.99):
        assert abs(recalibrar(p, 1.0) - p) < 1e-12


def test_recalibrar_reduce():
    """Con r < 1 la probabilidad SIEMPRE baja: se quitaron negativos, no positivos."""
    for p in (0.05, 0.3, 0.5, 0.9):
        assert recalibrar(p, 0.1) < p


def test_recalibrar_recupera_la_prevalencia():
    """El caso cerrado que hace la prueba útil: si la prevalencia real es 5% y se
    conserva 1 de cada 10 negativos, la prevalencia de la muestra es
    0.05 / (0.05 + 0.95*0.1) = 0.3448. Recalibrar ESE número tiene que devolver 0.05."""
    p_real, r = 0.05, 0.1
    p_muestra = p_real / (p_real + (1 - p_real) * r)
    assert abs(recalibrar(p_muestra, r) - p_real) < 1e-12


def test_regla_de_imputacion_usa_evidencia_y_no_un_umbral_fijo():
    """El caso real que rompió la primera versión de la regla.

    El grupo sin perfil tiene 5,335% de CTR contra 5,132% del grupo con perfil:
    una razón de 1,040, por debajo del umbral fijo del 5% que la versión anterior
    usaba, y sin embargo sobre 1,5M de observaciones la diferencia tiene z = 11.
    Con el umbral viejo esas columnas salían marcadas "ausencia no informativa",
    contradiciendo el hallazgo de la Fase 1. Se prueban las dos direcciones:
    diferencia chica pero real -> informativa; diferencia grande pero ruidosa -> no.
    """
    import pandas as pd

    from adbd.silver import _z_dos_proporciones, decidir_imputacion

    z = _z_dos_proporciones(81546, 1528526, 1284559, 25029435)
    assert z is not None and z > 10, z

    perfil = pd.DataFrame([
        {"columna": "real_pero_chica", "pct_nulos_en_cruce": 5.76,
         "impresiones_sin_dato": 1528526, "ctr_sin_dato_pct": 5.335,
         "ctr_con_dato_pct": 5.132, "razon_ctr": 1.04, "z": z},
        {"columna": "grande_pero_ruidosa", "pct_nulos_en_cruce": 3.0,
         "impresiones_sin_dato": 500, "ctr_sin_dato_pct": 6.0,
         "ctr_con_dato_pct": 5.1, "razon_ctr": 1.176, "z": 0.9},
    ])
    d = decidir_imputacion(perfil).set_index("columna")
    assert d.loc["real_pero_chica", "decision"] == "categoria_desconocido"
    assert "ES un dato" in d.loc["real_pero_chica", "motivo"]
    assert d.loc["grande_pero_ruidosa", "decision"] == "imputar_moda"


def test_recalibrar_monotona():
    """El orden se conserva: por eso AUC-ROC es invariante a esta corrección."""
    ps = [0.1, 0.2, 0.35, 0.6, 0.8]
    cal = [recalibrar(p, 0.07) for p in ps]
    assert cal == sorted(cal)


# ---------------------------------------------------------------------------
# 2. Pipeline completo sobre datos sintéticos.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def entorno():
    pytest.importorskip("pyspark")
    tmp = tempfile.mkdtemp(prefix="adbd_humo_")
    cwd = os.getcwd()
    try:
        from scripts.generar_datos_sinteticos import generar

        csv = os.path.join(tmp, "datos_csv")
        generar(csv, n_impresiones=60_000, n_usuarios=4_000, n_avisos=1_200, semilla=7)
        os.chdir(tmp)

        from adbd import bronze, config, gold as gold_mod, silver as silver_mod, utilidades

        config.RUTA_CSV = csv
        spark = utilidades.crear_sesion("ADBD-humo")
        bronze.construir(spark, rehacer=True)
        silver = silver_mod.construir(spark)
        oro = gold_mod.construir(spark, silver, escribir=False)
        yield spark, silver, oro
        spark.stop()
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)


def test_contrato_sanea_el_encabezado(entorno):
    """'new_user_class_level ' con espacio final tiene que llegar limpio a Silver."""
    _, silver, _ = entorno
    assert "new_user_class_level" in silver.columns
    assert not any(c.endswith(" ") for c in silver.columns)


def test_brand_no_es_predictor(entorno):
    """brand quedó descartada; solo sobrevive el flag."""
    _, silver, oro = entorno
    assert "brand" not in silver.columns
    assert "brand" not in oro.columns
    assert "brand_conocida" in oro.columns


def test_split_deja_el_burnin_fuera(entorno):
    from adbd import config, gold as gold_mod

    _, _, oro = entorno
    resumen = gold_mod.resumen_split(oro)
    assert (resumen.split == "burn-in (no se entrena)").sum() == 1
    train, val, test = gold_mod.particionar(oro)
    fechas_train = {str(r[0]) for r in train.select("fecha_local").distinct().collect()}
    assert config.DIA_BURNIN not in fechas_train
    assert len(fechas_train) == 5
    assert val.count() > 0 and test.count() > 0


def test_sin_nulos_en_los_features(entorno):
    """Un null que llega al VectorAssembler aborta el ajuste. Se verifica antes."""
    from pyspark.sql import functions as F

    from adbd import gold as gold_mod

    _, _, oro = entorno
    train, _, _ = gold_mod.particionar(oro)
    columnas = gold_mod.CATEGORICAS + gold_mod.NUMERICAS
    conteo = train.select(
        [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in columnas]
    ).collect()[0].asDict()
    con_nulos = {k: v for k, v in conteo.items() if v}
    assert not con_nulos, f"features con nulos: {con_nulos}"


def test_control_de_fuga(entorno):
    from adbd import gold as gold_mod

    spark, _, oro = entorno
    fuga = gold_mod.verificar_fuga(oro, spark)
    assert (fuga.veredicto == "OK").all(), fuga.to_string(index=False)


def test_submuestreo_conserva_todos_los_positivos(entorno):
    from pyspark.sql import functions as F

    from adbd import gold as gold_mod, modelos

    _, _, oro = entorno
    train, _, _ = gold_mod.particionar(oro)
    pos_antes = train.filter(F.col("clk") == 1).count()
    muestra, info = modelos.submuestrear_negativos(train, 0.1)
    assert info["positivos"] == pos_antes
    assert 0.05 < info["tasa_efectiva"] < 0.15
    assert info["prevalencia_muestra"] > info["prevalencia_real"]


def test_pipeline_ajusta_y_predice(entorno):
    """El camino completo: features -> LR -> recalibración -> métricas."""
    from adbd import evaluacion, features, gold as gold_mod, modelos
    from adbd.transformadores import CorrectorPrior

    _, _, oro = entorno
    train, val, _ = gold_mod.particionar(oro)
    muestra, info = modelos.submuestrear_negativos(train, 0.2)
    spec = modelos.catalogo_experimentos()["lr_completo"]
    pipe = features.pipeline(spec["estimador"], escalar=True)
    modelo = pipe.fit(muestra)

    corrector = CorrectorPrior(tasaNegativos=info["tasa_efectiva"])
    pred = corrector.transform(modelo.transform(val))
    m = evaluacion.metricas_binarias(pred)

    assert 0 <= m["auc_roc"] <= 1
    assert m["auc_pr"] >= 0
    # La recalibración tiene que dejar el CTR medio predicho en el orden de
    # magnitud del observado. Sin ella estaría ~5x arriba.
    assert 0.3 < m["sesgo_calibracion"] < 3.0, m
    # Y los nombres de las variables tienen que alinearse con el vector.
    imp = features.importancias(modelo)
    assert not imp.empty and imp.variable.notna().all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-v", "-x"]))
