"""
Fase 2 de punta a punta, sin notebook. Es la prueba de reproducibilidad que la
Fase 3 exige: desde los CSV crudos hasta las métricas y el tracking de MLflow,
con un solo comando y sin intervención manual.

    python -m scripts.correr_fase2 --ruta-csv ./datos_csv

Opciones útiles:
    --rapido         grids mínimos y ALS corto: para verificar que todo corre
    --sin-als        omite el experimento de recomendación
    --fraccion 0.2   trabaja sobre una fracción del dataset (pruebas)
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from pyspark.sql import functions as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adbd import bronze, config, evaluacion, features, gold as gold_mod, modelos  # noqa: E402
from adbd import seguimiento, silver as silver_mod, utilidades  # noqa: E402
from adbd.transformadores import CorrectorPrior  # noqa: E402
from pyspark.ml.functions import vector_to_array  # noqa: E402


def main(args):
    crono = utilidades.Crono()
    RES: dict = {}
    os.makedirs(config.RUTA_ARTEFACTOS, exist_ok=True)

    spark = utilidades.crear_sesion("ADBD-Fase2")
    RES["entorno"] = utilidades.huella_entorno(spark)
    print(RES["entorno"])

    # ---------------- BRONZE ----------------
    with crono.medir("bronze"):
        RES["bronze"] = bronze.construir(spark, rehacer=args.rehacer_bronze)
    imp, _, _ = bronze.leer(spark)
    RES["filas_bronze"] = imp.count()

    # ---------------- SILVER ----------------
    with crono.medir("perfilamiento de calidad"):
        perfil = silver_mod.perfilar_calidad(spark)
        decisiones = silver_mod.decidir_imputacion(perfil)
        RES["calidad"] = decisiones.to_dict("records")
        print(decisiones.to_string(index=False))

    with crono.medir("silver"):
        card = silver_mod.validar_cardinalidad(spark, RES["filas_bronze"])
        RES["cardinalidad"] = card.to_dict("records")
        silver = silver_mod.construir(spark, decisiones)
        if args.fraccion < 1:
            silver = silver.sample(False, args.fraccion, seed=config.SEMILLA)
        RES["filas_silver"] = silver.count()

    # ---------------- GOLD ----------------
    with crono.medir("gold"):
        gold = gold_mod.construir(spark, silver, escribir=not args.sin_escribir)
        RES["filas_gold"] = gold.count()
        RES["split"] = gold_mod.resumen_split(gold).to_dict("records")
        print(pd.DataFrame(RES["split"]).to_string(index=False))

    with crono.medir("control de fuga"):
        fuga = gold_mod.verificar_fuga(gold, spark)
        RES["fuga"] = fuga.to_dict("records")
        print(fuga.to_string(index=False))
        assert (fuga.veredicto == "OK").all(), "control de fuga NO superado; no se sigue"

    RES["registro_variables"] = gold_mod.registro_variables().to_dict("records")

    train, val, test = gold_mod.particionar(gold)
    train.cache(); val.cache(); test.cache()
    RES["n_train"], RES["n_val"], RES["n_test"] = train.count(), val.count(), test.count()

    # ---------------- Desbalance ----------------
    with crono.medir("submuestreo de negativos"):
        train_ds, info_ds = modelos.submuestrear_negativos(train, args.tasa_negativos)
        train_ds = train_ds.cache()
        train_ds.count()
        RES["submuestreo"] = info_ds
        print(info_ds)
    r_efectiva = info_ds["tasa_efectiva"]
    corrector = CorrectorPrior(inputCol="probability", outputCol="p_calibrada",
                               tasaNegativos=r_efectiva)

    # ---------------- MLflow ----------------
    seguimiento.iniciar(args.experimento)
    params_datos = seguimiento.parametros_datos(info_ds)

    # Línea base
    p0, base_test = modelos.linea_base_prior(train, test)
    m_base = evaluacion.metricas_binarias(base_test)
    RES["linea_base"] = {"p0": p0, **m_base}
    seguimiento.registrar_run(
        "baseline_prior",
        "¿Cuál es el piso real? Predecir siempre el CTR histórico.",
        {**params_datos, "modelo": "constante", "p0": round(p0, 6)},
        {f"test_{k}": v for k, v in m_base.items() if isinstance(v, (int, float))},
    )
    print(f"línea base · AUC-PR {m_base['auc_pr']:.5f} (= prevalencia {m_base['ctr_observado']:.5f})")

    # Experimentos supervisados
    numericas_sin_historia = [c for c in gold_mod.NUMERICAS
                              if not c.startswith(("ctr_hist_", "log_imp_hist_", "ctr_usuario",
                                                   "log_imp_previas", "ctr_movil"))]
    catalogo = modelos.catalogo_experimentos(numericas_sin_historia)
    if args.rapido:
        for spec in catalogo.values():
            spec["grid"] = spec["grid"][:1]
    if args.solo:
        catalogo = {k: v for k, v in catalogo.items() if k in args.solo.split(",")}

    resultados, curvas = {}, {}
    for nombre, spec in catalogo.items():
        print(f"\n===== {nombre} · {spec['pregunta']}")
        with crono.medir(f"experimento {nombre}"):
            modelo, info_cv = modelos.ajustar_con_cv(
                train_ds, spec, folds=2 if args.rapido else 3,
                fraccion_busqueda=1.0 if args.rapido else config.FRACCION_BUSQUEDA)

        pv = corrector.transform(modelo.transform(val))
        pt = corrector.transform(modelo.transform(test))
        m_val = evaluacion.metricas_binarias(pv)
        m_test = evaluacion.metricas_binarias(pt)
        deciles = evaluacion.tabla_deciles(pt)
        curvas[nombre] = evaluacion.curva_pr(pt)
        lift_d1 = float(deciles.iloc[0]["lift"])

        # Brecha de optimismo del k-fold aleatorio, medida COMO CORRESPONDE.
        # El CrossValidator calcula su AUC-PR sobre pliegues ya submuestreados
        # (prevalencia ~31%) y la validación va sobre el día completo (~5%).
        # Restar esos dos números da una "brecha" enorme que no mide optimismo
        # sino la diferencia de prevalencia: AUC-PR depende de ella por definición.
        # La comparación válida es contra una validación submuestreada a la MISMA
        # tasa; el contraste en AUC-ROC —invariante al submuestreo— queda como
        # control cruzado.
        val_ds, _ = modelos.submuestrear_negativos(val, r_efectiva)
        m_val_ds = evaluacion.metricas_binarias(corrector.transform(modelo.transform(val_ds)))
        brecha = info_cv["cv_mejor"] - m_val_ds["auc_pr"]
        brecha_roc = m_val_ds["auc_roc"] - m_val["auc_roc"]

        metricas = (
            {f"val_{k}": v for k, v in m_val.items() if isinstance(v, (int, float))}
            | {f"test_{k}": v for k, v in m_test.items() if isinstance(v, (int, float))}
            | {"test_lift_d1": lift_d1,
               "cv_auc_pr": info_cv["cv_mejor"],
               "val_ds_auc_pr": m_val_ds["auc_pr"],
               "brecha_optimismo_cv_val": brecha,
               "brecha_auc_roc_muestra_vs_completo": brecha_roc,
               "t_busqueda_s": info_cv["t_busqueda_s"],
               "t_ajuste_final_s": info_cv["t_ajuste_final_s"]}
        )
        run_id = seguimiento.registrar_run(
            nombre, spec["pregunta"],
            {**params_datos, **info_cv["mejores_parametros"],
             "estimador": type(spec["estimador"]).__name__,
             "escalado": spec.get("escalar", False),
             "n_combinaciones": info_cv["n_combinaciones"],
             "folds": info_cv["folds"],
             "fraccion_busqueda": info_cv["fraccion_busqueda"]},
            metricas,
            tablas={"deciles": deciles, "curva_pr": curvas[nombre]},
            modelo=modelo if args.registrar_modelos else None,
        )
        imp = features.importancias(modelo, numericas=spec.get("numericas"))
        if not imp.empty:
            imp.to_csv(os.path.join(config.RUTA_ARTEFACTOS, f"{nombre}__importancias.csv"),
                       index=False)
        resultados[nombre] = {"run_id": run_id, "cv": info_cv, "val": m_val, "test": m_test,
                              "importancias": imp.to_dict("records"),
                              "val_submuestreada": m_val_ds, "lift_d1": lift_d1,
                              "brecha_optimismo": brecha, "brecha_auc_roc": brecha_roc,
                              "modelo": modelo, "spec": nombre,
                              "deciles": deciles.to_dict("records")}
        print(f"  AUC-PR test {m_test['auc_pr']:.5f} · AUC-ROC {m_test['auc_roc']:.5f} "
              f"· lift D1 {lift_d1:.2f}x · brecha CV-val (like-for-like) {brecha:+.5f}")

    # ---------------- Contraste: weightCol sobre el train COMPLETO ----------------
    # Mide lo que el atajo cuesta de verdad. Sin esto, "submuestreamos por costo"
    # es una afirmación sin número detrás.
    if args.con_pesos and "lr_completo" in catalogo:
        print("\n===== lr_pesos_completo (sin submuestrear, con weightCol)")
        with crono.medir("experimento lr_pesos_completo"):
            spec = dict(catalogo["lr_completo"])
            train_w, info_w = modelos.agregar_pesos(train)
            est = spec["estimador"].copy()
            est.setWeightCol("peso")
            pipe_w = features.pipeline(est, numericas=spec.get("numericas"),
                                       escalar=spec.get("escalar", False))
            import time as _t
            t0 = _t.perf_counter()
            modelo_w = pipe_w.fit(train_w)
            t_w = round(_t.perf_counter() - t0, 1)
        # weightCol tampoco deja la probabilidad calibrada: pesar los positivos por
        # w multiplica sus odds por w, igual que el submuestreo las multiplica por 1/r.
        # Es la MISMA corrección con r = 1/w. Omitirla sería el error que este
        # experimento existe para desmentir.
        corrector_w = CorrectorPrior(inputCol="probability", outputCol="p_calibrada",
                                     tasaNegativos=1.0 / info_w["peso_positivo"])
        pt_w = corrector_w.transform(modelo_w.transform(test))
        m_test_w = evaluacion.metricas_binarias(pt_w)
        dec_w = evaluacion.tabla_deciles(pt_w)
        RES["contraste_pesos"] = {
            **info_w, "t_ajuste_final_s": t_w, "test": m_test_w,
            "lift_d1": float(dec_w.iloc[0]["lift"]),
            "delta_auc_pr_vs_submuestreo":
                round(m_test_w["auc_pr"] - resultados["lr_completo"]["test"]["auc_pr"], 6),
            "razon_tiempo": round(t_w / max(resultados["lr_completo"]["cv"]["t_ajuste_final_s"], 1e-9), 2),
        }
        seguimiento.registrar_run(
            "lr_pesos_completo",
            "¿Cuánto AUC-PR cuesta el submuestreo frente a entrenar con el dataset completo?",
            {**params_datos, "estimador": "LogisticRegression", "weightCol": "peso",
             "submuestreo": "no", "peso_positivo": info_w["peso_positivo"]},
            {f"test_{k}": v for k, v in m_test_w.items() if isinstance(v, (int, float))}
            | {"t_ajuste_final_s": t_w, "test_lift_d1": float(dec_w.iloc[0]["lift"])},
            tablas={"deciles": dec_w},
        )
        print(f"  AUC-PR test {m_test_w['auc_pr']:.5f} en {t_w} s "
              f"(submuestreado: {resultados['lr_completo']['test']['auc_pr']:.5f} en "
              f"{resultados['lr_completo']['cv']['t_ajuste_final_s']} s)")

    # ---------------- Escalamiento: ¿más datos o más modelo? ----------------
    if args.curva_aprendizaje and resultados:
        mejor_nombre = max(resultados, key=lambda k: resultados[k]["test"]["auc_pr"])
        print(f"\n===== curva de aprendizaje · {mejor_nombre}")
        with crono.medir("curva de aprendizaje"):
            d = modelos.curva_aprendizaje(
                train_ds, catalogo[mejor_nombre],
                fracciones=(0.25, 1.0) if args.rapido else (0.1, 0.25, 0.5, 1.0),
                evaluar=val, corrector=corrector,
                metrica_fn=evaluacion.metricas_binarias)
        RES["curva_aprendizaje"] = d.to_dict("records")
        evaluacion.graficar_curva_aprendizaje(
            d, os.path.join(config.RUTA_ARTEFACTOS, "curva_aprendizaje.png"))

    # ---------------- La demostración de por qué se evalúa sin submuestrear ------
    if resultados:
        mejor_nombre = max(resultados, key=lambda k: resultados[k]["test"]["auc_pr"])
        m = resultados[mejor_nombre]["modelo"]
        test_ds, _ = modelos.submuestrear_negativos(test, r_efectiva)
        RES["efecto_submuestreo_en_la_evaluacion"] = evaluacion.comparar_submuestreo(
            corrector.transform(m.transform(test)),
            corrector.transform(m.transform(test_ds)),
        ).to_dict("records")

    for r in resultados.values():
        r.pop("modelo", None)
    RES["experimentos"] = resultados

    # ---------------- ALS ----------------
    if not args.sin_als:
        print("\n===== als_implicito")
        with crono.medir("ALS implícito"):
            m_train = modelos.matriz_implicita(train).cache()
            m_test = modelos.matriz_implicita(test).cache()
            modelos.verificar_ids_als(m_train, m_test)
            als_modelo, t_als = modelos.entrenar_als(
                m_train, iters=3 if args.rapido else None)

            recs = (als_modelo.recommendForAllUsers(config.TOP_K)
                    .select(F.col("userid").cast("long").alias("userid"),
                            F.col("recommendations.cate_id").alias("recomendadas")))
            reales = (m_test.groupBy("userid")
                      .agg(F.collect_set(F.col("cate_id").cast("int")).alias("reales")))
            m_als = evaluacion.metricas_ranking(recs, reales)

            top = modelos.recomendaciones_populares(m_train)
            pop = reales.select("userid").withColumn(
                "recomendadas", F.array(*[F.lit(int(c)) for c in top]))
            m_pop = evaluacion.metricas_ranking(pop, reales)

        RES["als"] = {"als": m_als, "popularidad": m_pop, "t_ajuste_s": t_als,
                      "rank": config.ALS_RANK, "alpha": config.ALS_ALPHA}
        seguimiento.registrar_run(
            "als_implicito",
            "¿Un modelo de recomendación bate a la popularidad en exposición por categoría?",
            {**params_datos, "modelo": "ALS", "rank": config.ALS_RANK,
             "regParam": config.ALS_REG, "alpha": config.ALS_ALPHA, "implicitPrefs": True},
            {f"test_{k}": v for k, v in m_als.items() if isinstance(v, (int, float))}
            | {f"pop_{k}": v for k, v in m_pop.items() if isinstance(v, (int, float))}
            | {"t_ajuste_final_s": t_als},
        )
        print(f"  ALS  MAP@{config.TOP_K} {m_als['map_at_k']:.5f} · NDCG {m_als['ndcg_at_k']:.5f}")
        print(f"  pop. MAP@{config.TOP_K} {m_pop['map_at_k']:.5f} · NDCG {m_pop['ndcg_at_k']:.5f}")

    # ---------------- Artefactos y cierre ----------------
    if curvas:
        prevalencia = RES["linea_base"]["ctr_observado"]
        evaluacion.graficar_pr(curvas, prevalencia,
                               os.path.join(config.RUTA_ARTEFACTOS, "curva_pr.png"))
        mejor = max(resultados, key=lambda k: resultados[k]["test"]["auc_pr"])
        evaluacion.graficar_calibracion(
            pd.DataFrame(resultados[mejor]["deciles"]),
            os.path.join(config.RUTA_ARTEFACTOS, "calibracion.png"))
        RES["mejor_experimento"] = mejor

    RES["comparativa_mlflow"] = seguimiento.tabla_comparativa(args.experimento).to_dict("records")
    RES["tiempos"] = crono.marcas
    RES["t_total_s"] = crono.total()
    RES["usd_referencial"] = utilidades.usd(crono.total())
    utilidades.exportar(RES, args.salida_json)
    print(f"\nTOTAL {RES['t_total_s']} s · US$ {RES['usd_referencial']} de cómputo equivalente")
    spark.stop()
    return RES


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ruta-csv", default=None)
    ap.add_argument("--experimento", default=config.NOMBRE_EXPERIMENTO)
    ap.add_argument("--tasa-negativos", type=float, default=config.TASA_NEGATIVOS)
    ap.add_argument("--fraccion", type=float, default=1.0)
    ap.add_argument("--salida-json", default="resultados_fase2.json")
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--sin-als", action="store_true")
    ap.add_argument("--sin-escribir", action="store_true")
    ap.add_argument("--rehacer-bronze", action="store_true")
    ap.add_argument("--registrar-modelos", action="store_true")
    ap.add_argument("--con-pesos", action="store_true",
                    help="entrena además con weightCol sobre el train completo (contraste)")
    ap.add_argument("--curva-aprendizaje", action="store_true",
                    help="ajusta el mejor modelo sobre fracciones crecientes del train")
    ap.add_argument("--solo", default=None)
    a = ap.parse_args()
    if a.ruta_csv:
        config.RUTA_CSV = a.ruta_csv
    main(a)
