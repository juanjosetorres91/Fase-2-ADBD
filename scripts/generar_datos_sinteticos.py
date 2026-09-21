"""
Genera CSV sintéticos con el MISMO esquema —y los mismos defectos— que el espejo
de Kaggle, para poder correr el pipeline completo en segundos.

No es un juguete: reproduce a propósito las tres cosas que la Fase 1 encontró en
los datos reales, porque son las que el pipeline tiene que sobrevivir.
  1. El encabezado 'new_user_class_level ' con un espacio final.
  2. Valores no numéricos en `brand`, que el cast convierte en nulos inexistentes.
  3. Usuarios con impresiones pero SIN fila en user_profile (el 5,76% del cruce).
Además mantiene el desbalance (~5% de CTR) y la ventana de 8 días locales, que es
lo que hace que el split temporal y el burn-in sean ejercitables.

Uso:  python -m scripts.generar_datos_sinteticos --salida ./datos_csv --impresiones 300000
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import random

# 2017-05-05 16:00:00 UTC == 2017-05-06 00:00:00 en UTC+8 (hora local de Hangzhou)
INICIO_UTC = dt.datetime(2017, 5, 5, 16, 0, 0, tzinfo=dt.timezone.utc)
DIAS = 8
PIDS = ["430539_1007", "430548_1007"]


def generar(salida: str, n_impresiones: int, n_usuarios: int, n_avisos: int, semilla: int):
    rnd = random.Random(semilla)
    os.makedirs(salida, exist_ok=True)
    ts0 = int(INICIO_UTC.timestamp())
    ts1 = ts0 + DIAS * 86400 - 1

    # --- ad_feature -------------------------------------------------------
    n_cate, n_camp, n_cust = 40, 400, 250
    avisos = []
    for a in range(1, n_avisos + 1):
        cate = rnd.randint(1, n_cate)
        avisos.append({
            "adgroup_id": a,
            "cate_id": cate,
            "campaign_id": rnd.randint(1, n_camp),
            "customer": rnd.randint(1, n_cust),
            # 3% de valores no numéricos: el cast los convierte en nulos que NO
            # existen en el origen. Es el defecto real de ad_feature.brand.
            "brand": ("NULL" if rnd.random() < 0.03 else rnd.randint(1, 5000)),
            "price": round(abs(rnd.lognormvariate(5.5, 1.1)), 2) * (0 if rnd.random() < 0.01 else 1),
        })
    with open(os.path.join(salida, "ad_feature.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(avisos[0]))
        w.writeheader(); w.writerows(avisos)

    # --- user_profile -----------------------------------------------------
    # Solo el 94% de los usuarios tiene perfil: el 6% restante reproduce el
    # segmento "sin_perfil" que la Fase 1 decidió conservar con flag.
    con_perfil = set(rnd.sample(range(1, n_usuarios + 1), int(n_usuarios * 0.94)))
    encabezado = ["userid", "cms_segid", "cms_group_id", "final_gender_code", "age_level",
                  "pvalue_level", "shopping_level", "occupation", "new_user_class_level "]
    with open(os.path.join(salida, "user_profile.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(encabezado)                      # <- el espacio final va a propósito
        for u in sorted(con_perfil):
            w.writerow([
                u, rnd.randint(0, 96), rnd.randint(1, 13),
                rnd.choice([1, 2]), rnd.randint(0, 6),
                "" if rnd.random() < 0.54 else rnd.randint(1, 3),      # 54% nulos
                rnd.randint(1, 3), rnd.choice([0, 1]),
                "" if rnd.random() < 0.32 else rnd.randint(1, 4),      # 32% nulos
            ])

    # --- raw_sample -------------------------------------------------------
    # El CTR depende de señales reales para que el modelo tenga algo que aprender:
    # calidad del aviso, posición (pid), hora local y fatiga por usuario.
    calidad = {a["adgroup_id"]: rnd.betavariate(2, 30) for a in avisos}
    sesgo_pid = {PIDS[0]: 1.12, PIDS[1]: 0.92}
    vistas: dict[int, int] = {}
    ruta = os.path.join(salida, "raw_sample.csv")
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user", "time_stamp", "adgroup_id", "pid", "nonclk", "clk"])
        for _ in range(n_impresiones):
            u = rnd.randint(1, n_usuarios)
            ts = rnd.randint(ts0, ts1)
            a = rnd.randint(1, n_avisos)
            pid = rnd.choice(PIDS)
            hora_local = ((ts + 8 * 3600) // 3600) % 24
            factor_hora = 1.25 if 19 <= hora_local <= 23 else (0.8 if hora_local < 7 else 1.0)
            n_vistas = vistas.get(u, 0)
            vistas[u] = n_vistas + 1
            fatiga = 1.0 / (1.0 + 0.05 * min(n_vistas, 20))     # el hallazgo W2
            p = min(0.9, calidad[a] * sesgo_pid[pid] * factor_hora * fatiga)
            clk = 1 if rnd.random() < p else 0
            w.writerow([u, ts, a, pid, 1 - clk, clk])

    print(f"CSV sintéticos en {salida}:")
    for n in ("raw_sample.csv", "ad_feature.csv", "user_profile.csv"):
        p = os.path.join(salida, n)
        print(f"  {n:<20} {os.path.getsize(p)/1e6:7.2f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="./datos_csv")
    ap.add_argument("--impresiones", type=int, default=300_000)
    ap.add_argument("--usuarios", type=int, default=12_000)
    ap.add_argument("--avisos", type=int, default=4_000)
    ap.add_argument("--semilla", type=int, default=7)
    a = ap.parse_args()
    generar(a.salida, a.impresiones, a.usuarios, a.avisos, a.semilla)
