"""
MLflow. Backend de archivos local (`file:./mlruns`), que es lo que funciona en
Colab gratuito sin servidor ni base de datos, y se entrega comprimido junto con
el repositorio para que el tracking sea auditable y no una captura de pantalla.

Qué se registra en cada run, y por qué cada cosa:
  · parámetros  — los del modelo Y los del pipeline de datos (tasa de submuestreo,
                  m de suavizado, fechas del split). Sin estos últimos, dos runs
                  con el mismo AUC son indistinguibles y no se sabe cuál reproducir.
  · métricas    — sobre validación y sobre test, con el sufijo del conjunto.
  · artefactos  — curvas PR, calibración, tabla de deciles, importancias,
                  el registro de variables y el modelo del ganador.
  · etiquetas   — la pregunta que el experimento responde, para que la tabla
                  comparativa se lea sin tener que abrir el código.
"""
from __future__ import annotations

import json
import os

import mlflow

from . import config


def iniciar(nombre_experimento: str | None = None, ruta: str | None = None) -> str:
    ruta = ruta or config.RUTA_MLRUNS
    os.makedirs(ruta, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{os.path.abspath(ruta)}")
    exp = nombre_experimento or config.NOMBRE_EXPERIMENTO
    mlflow.set_experiment(exp)
    print(f"MLflow · tracking en {mlflow.get_tracking_uri()} · experimento '{exp}'")
    return exp


def parametros_datos(info_muestra: dict, extra: dict | None = None) -> dict:
    """Los parámetros del PIPELINE DE DATOS, que son los que de verdad hacen
    reproducible un run. El modelo es la parte fácil de repetir."""
    p = {
        "split_burnin": config.DIA_BURNIN,
        "split_train": f"{config.TRAIN_INI}..{config.TRAIN_FIN}",
        "split_val": config.DIA_VAL,
        "split_test": config.DIA_TEST,
        "m_suavizado": config.M_SUAVIZADO,
        "m_suavizado_usuario": config.M_SUAVIZADO_USUARIO,
        "entidades_historicas": ",".join(config.ENTIDADES_HISTORICAS),
        "tasa_negativos_nominal": info_muestra.get("tasa_nominal"),
        "tasa_negativos_efectiva": round(info_muestra.get("tasa_efectiva", 1.0), 6),
        "filas_entrenamiento": info_muestra.get("filas_entrenamiento"),
        "prevalencia_muestra": round(info_muestra.get("prevalencia_muestra", 0), 6),
        "prevalencia_real": round(info_muestra.get("prevalencia_real", 0), 6),
        "semilla": config.SEMILLA,
    }
    p.update(extra or {})
    return p


def registrar_run(nombre, pregunta, parametros, metricas, artefactos=None,
                  tablas=None, modelo=None, ruta_artefactos=None):
    """Un run completo. Devuelve el run_id para poder citarlo en el informe."""
    ruta_artefactos = ruta_artefactos or config.RUTA_ARTEFACTOS
    os.makedirs(ruta_artefactos, exist_ok=True)
    with mlflow.start_run(run_name=nombre) as run:
        mlflow.set_tag("pregunta", pregunta)
        mlflow.set_tag("fase", "2")
        mlflow.log_params({k: v for k, v in parametros.items() if v is not None})
        mlflow.log_metrics({k: float(v) for k, v in metricas.items()
                            if v is not None and isinstance(v, (int, float))})
        for a in (artefactos or []):
            if a and os.path.exists(a):
                mlflow.log_artifact(a)
        for nombre_tabla, df in (tablas or {}).items():
            ruta = os.path.join(ruta_artefactos, f"{nombre}__{nombre_tabla}.csv")
            df.to_csv(ruta, index=False)
            mlflow.log_artifact(ruta)
        if modelo is not None:
            try:
                from mlflow import spark as mlflow_spark

                mlflow_spark.log_model(modelo, artifact_path="modelo")
            except Exception as e:   # en Colab gratuito el guardado puede quedarse sin disco
                mlflow.set_tag("modelo_no_registrado", f"{type(e).__name__}: {str(e)[:150]}")
        return run.info.run_id


def tabla_comparativa(experimento: str | None = None):
    """La comparación de experimentos que pide la pauta, leída DESDE MLflow y no
    desde variables en memoria: si la tabla del informe sale del tracking, el
    tracking es la fuente de verdad y no una decoración."""
    import pandas as pd

    exp = mlflow.get_experiment_by_name(experimento or config.NOMBRE_EXPERIMENTO)
    if exp is None:
        return pd.DataFrame()
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    if runs.empty:
        return runs
    cols = ["tags.mlflow.runName", "tags.pregunta",
            "metrics.test_auc_pr", "metrics.test_auc_roc", "metrics.test_logloss",
            "metrics.test_lift_d1", "metrics.val_auc_pr",
            "metrics.t_ajuste_final_s", "metrics.t_busqueda_s", "run_id"]
    presentes = [c for c in cols if c in runs.columns]
    d = runs[presentes].copy()
    d.columns = [c.replace("tags.mlflow.runName", "experimento")
                  .replace("tags.", "").replace("metrics.", "") for c in presentes]
    if "test_auc_pr" in d.columns:
        d = d.sort_values("test_auc_pr", ascending=False)
    return d.reset_index(drop=True)


def guardar_json(obj, ruta: str) -> str:
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    return ruta
