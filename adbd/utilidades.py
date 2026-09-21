"""Cronómetro, tamaños en disco y exportación de resultados."""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager

from . import config


class Crono:
    """Registro de tiempos con nombre. Reemplaza los %%time sueltos por algo que
    después se pueda tabular, que es lo que faltó en la v1 de la Fase 1."""

    def __init__(self) -> None:
        self.marcas: dict[str, float] = {}

    @contextmanager
    def medir(self, etiqueta: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.marcas[etiqueta] = round(time.perf_counter() - t0, 2)
            print(f"  ⏱  {etiqueta}: {self.marcas[etiqueta]} s")

    def total(self) -> float:
        return round(sum(self.marcas.values()), 2)

    def tabla(self):
        import pandas as pd

        return (
            pd.DataFrame([{"etapa": k, "segundos": v} for k, v in self.marcas.items()])
            .assign(
                minutos=lambda d: (d.segundos / 60).round(2),
                usd_referencial=lambda d: d.segundos.map(usd),
            )
            .sort_values("segundos", ascending=False)
            .reset_index(drop=True)
        )


def usd(segundos: float) -> float:
    """Valoriza tiempo de cómputo contra una VM de mercado. Referencia, no gasto."""
    return round(segundos / 3600 * config.TARIFA_USD_HORA, 4)


def tam_mb(ruta: str) -> float:
    if not os.path.exists(ruta):
        return 0.0
    if os.path.isfile(ruta):
        return os.path.getsize(ruta) / 1e6
    return sum(
        os.path.getsize(os.path.join(d, f))
        for d, _, fs in os.walk(ruta)
        for f in fs
    ) / 1e6


def limpiar(o):
    """numpy/pandas -> tipos JSON. Sin esto json.dump revienta con np.int64.

    NaN e infinitos se convierten en null: `json.dump` los escribe como `NaN`,
    que Python vuelve a leer pero **no es JSON válido** — cualquier otro lector
    (el generador del informe, por ejemplo) falla al parsear el archivo.
    """
    import math

    import numpy as np

    if isinstance(o, dict):
        return {str(k): limpiar(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [limpiar(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def exportar(res: dict, ruta: str = "resultados_fase2.json") -> str:
    res = dict(res)
    res["generado"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(limpiar(res), f, ensure_ascii=False, indent=2, default=str,
                  allow_nan=False)
    print(f"{ruta} escrito · {len(res)} claves")
    return ruta


def preparar_hadoop_windows(verboso: bool = True) -> str | None:
    """En Windows, Spark resuelve permisos del sistema de archivos local con
    `winutils.exe` + `hadoop.dll`, que el paquete `pyspark` NO empaqueta. Sin ellos
    la JVM muere antes de crear el directorio temporal del driver:

        java.io.FileNotFoundException: HADOOP_HOME and hadoop.home.dir are unset

    No es un error del pipeline —el mismo código corre tal cual en Colab y en
    Linux— sino un requisito del entorno, y por eso se resuelve aquí una sola vez
    en lugar de repetirlo en el notebook, en los tests y en el script de CLI.

    Fuera de Windows no hace nada. Devuelve el HADOOP_HOME efectivo, o None.
    """
    import platform

    if platform.system() != "Windows":
        return None

    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidatos = [
        os.environ.get("HADOOP_HOME"),          # lo que el entorno ya declare manda
        os.path.join(raiz, "vendor", "hadoop"),  # copia versionada con el repositorio
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "hadoop"),
        os.path.join(os.path.expanduser("~"), "hadoop"),
        r"C:\hadoop",
    ]
    for c in candidatos:
        if not c or not os.path.isfile(os.path.join(c, "bin", "winutils.exe")):
            continue
        binario = os.path.join(c, "bin")
        os.environ["HADOOP_HOME"] = c
        # hadoop.dll se carga por java.library.path, que en Windows incluye PATH.
        rutas = os.environ.get("PATH", "").split(os.pathsep)
        if not any(os.path.normcase(r) == os.path.normcase(binario) for r in rutas):
            os.environ["PATH"] = binario + os.pathsep + os.environ.get("PATH", "")
        if verboso:
            print(f"  Windows · HADOOP_HOME = {c}")
        return c

    raise RuntimeError(
        "Windows sin los binarios nativos de Hadoop: Spark no puede arrancar.\n"
        "  Faltan winutils.exe y hadoop.dll (Hadoop 3.3.x, el que empaqueta "
        "pyspark 3.5.4).\n"
        f"  Déjalos en {os.path.join(raiz, 'vendor', 'hadoop', 'bin')} "
        "o exporta HADOOP_HOME apuntando a la carpeta que los contiene.\n"
        "  En Linux, macOS y Colab no hace falta nada de esto (ver README)."
    )


def preparar_java(verboso: bool = True) -> str | None:
    """Deja una JVM alcanzable para el lanzador de Spark, que la busca en
    `JAVA_HOME/bin/java` y, si no, en `java` a secas por el PATH.

    El caso que esto resuelve: el JDK está instalado DENTRO del entorno conda
    (`Library/lib/jvm` en Windows, `lib/jvm` en Linux/macOS) y solo entra al PATH
    al hacer `conda activate`. Un kernel de Jupyter lanzado por VSCode o por
    `jupyter lab` desde otro entorno arranca sin esa activación, no encuentra
    `java`, y Spark muere con un mensaje que no dice nada de Java:

        PySparkRuntimeError: [JAVA_GATEWAY_EXITED] Java gateway process exited
        before sending its port number.

    Orden: JAVA_HOME si ya está definido y es válido; `java` en el PATH; el JDK
    del entorno del intérprete actual. Si nada aparece, se detiene con el mensaje
    de qué instalar. Devuelve el JAVA_HOME efectivo, o None si se usa el del PATH.
    """
    import shutil
    import sys

    exe = "java.exe" if os.name == "nt" else "java"

    jh = os.environ.get("JAVA_HOME")
    if jh and os.path.isfile(os.path.join(jh, "bin", exe)):
        return jh
    if shutil.which("java"):
        return None

    candidatos = [
        os.path.join(sys.prefix, "Library", "lib", "jvm"),   # conda-forge openjdk, Windows
        os.path.join(sys.prefix, "lib", "jvm"),              # conda-forge openjdk, Linux/macOS
        os.path.join(sys.prefix, "Library"),                 # java.exe directo en Library/bin
        sys.prefix,
    ]
    for c in candidatos:
        if os.path.isfile(os.path.join(c, "bin", exe)):
            os.environ["JAVA_HOME"] = c
            os.environ["PATH"] = os.path.join(c, "bin") + os.pathsep + os.environ.get("PATH", "")
            if verboso:
                print(f"  JAVA_HOME = {c}")
            return c

    raise RuntimeError(
        "No hay una JVM alcanzable: ni JAVA_HOME, ni `java` en el PATH, ni un JDK en "
        f"el entorno {sys.prefix}.\n"
        "  Spark 3.5 necesita Java 11 o 17. En conda: conda install -c conda-forge openjdk=17\n"
        "  Con un JDK instalado aparte: exportar JAVA_HOME apuntando a su carpeta raíz."
    )


def crear_sesion(nombre: str = "ADBD-Fase2"):
    """Sesión de Spark con la configuración declarada en config.SPARK_CONF."""
    from pyspark.sql import SparkSession

    preparar_java()
    preparar_hadoop_windows()

    # Las banderas de módulo tienen que llegar a la JVM antes de que arranque.
    args = os.environ.get("PYSPARK_SUBMIT_ARGS", "")
    if "--add-opens" not in args:
        base = args.replace("pyspark-shell", "").strip()
        os.environ["PYSPARK_SUBMIT_ARGS"] = (
            f'{base} --driver-java-options "{config.OPCIONES_JVM}" pyspark-shell'
        ).strip()

    b = SparkSession.builder.appName(nombre)
    for k, v in config.SPARK_CONF.items():
        b = b.config(k, v)
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def huella_entorno(spark) -> dict:
    """Qué máquina produjo estos números. Sin esto ningún tiempo es interpretable."""
    import multiprocessing
    import platform

    try:
        import psutil

        ram = f"{round(psutil.virtual_memory().total / 1e9, 1)} GB"
    except Exception:
        ram = "no determinada"
    return {
        "so": platform.system(),
        "nucleos": multiprocessing.cpu_count(),
        "ram": ram,
        "spark": spark.version,
        "shuffle_partitions": spark.conf.get("spark.sql.shuffle.partitions"),
        "tz_sesion": spark.conf.get("spark.sql.session.timeZone"),
    }
