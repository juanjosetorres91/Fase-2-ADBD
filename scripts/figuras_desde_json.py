"""
Rehace las figuras del informe a partir de `resultados_fase2.json`, sin Spark.

Existe por una razón práctica: el informe se arma desde el JSON, pero las figuras
las escribía el notebook en `artefactos_fase2/`. Si alguien mueve el JSON a otra
máquina —o regenera el informe meses después— las figuras no viajan con él. Este
script las reconstruye de la misma fuente de verdad que las tablas, así que la
figura y la tabla de al lado no pueden discrepar.

La curva PR no se puede reconstruir desde el JSON (necesita los bins de score, que
no se exportan), así que esa sigue viniendo del notebook.

    python -m scripts.figuras_desde_json [resultados_fase2.json] [artefactos_fase2]
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def calibracion(deciles, ruta, titulo="Calibración por decil de score"):
    px = [d["p_media_pct"] for d in deciles]
    py = [d["ctr_pct"] for d in deciles]
    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=130)
    ax.plot(px, py, "o-", lw=1.5)
    lim = max(max(px), max(py)) * 1.08
    ax.plot([0, lim], [0, lim], ls="--", lw=1, color="#888", label="calibración perfecta")
    for d in deciles:
        if d["decil"] in ("D1", "D10"):
            ax.annotate(d["decil"], (d["p_media_pct"], d["ctr_pct"]),
                        textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.set_xlabel("CTR predicho medio (%)")
    ax.set_ylabel("CTR observado (%)")
    ax.set_title(titulo)
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)
    return ruta


def curva_aprendizaje(curva, ruta):
    filas = [c["filas"] for c in curva]
    auc = [c["auc_pr"] for c in curva]
    t = [c["t_ajuste_s"] for c in curva]
    fig, ax1 = plt.subplots(figsize=(6, 4.2), dpi=130)
    ax1.plot(filas, auc, "o-", color="#1f77b4", label="AUC-PR")
    ax1.set_xlabel("filas de entrenamiento")
    ax1.set_ylabel("AUC-PR", color="#1f77b4")
    # El eje hay que abrirlo o una curva plana se ve como una montaña.
    lo, hi = min(auc), max(auc)
    margen = max((hi - lo) * 4, hi * 0.02)
    ax1.set_ylim(lo - margen, hi + margen)
    ax2 = ax1.twinx()
    ax2.plot(filas, t, "s--", color="#d62728", label="tiempo de ajuste (s)")
    ax2.set_ylabel("segundos", color="#d62728")
    ax1.set_title("¿Más datos, o más modelo? Curva de aprendizaje y su costo")
    ax1.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)
    return ruta


def main(ruta_json="resultados_fase2.json", destino="artefactos_fase2"):
    with open(ruta_json, encoding="utf-8") as f:
        res = json.load(f)
    os.makedirs(destino, exist_ok=True)
    hechas = []

    mejor = res.get("mejor_experimento")
    deciles = (res.get("experimentos", {}).get(mejor, {}) or {}).get("deciles") or []
    if deciles:
        hechas.append(calibracion(deciles, os.path.join(destino, "calibracion.png")))
    if res.get("curva_aprendizaje"):
        hechas.append(curva_aprendizaje(res["curva_aprendizaje"],
                                        os.path.join(destino, "curva_aprendizaje.png")))
    for h in hechas:
        print("escrito", h)
    if not hechas:
        print("nada que rehacer: el JSON no trae deciles ni curva de aprendizaje")
    return hechas


if __name__ == "__main__":
    main(*(sys.argv[1:3] or ["resultados_fase2.json", "artefactos_fase2"]))
