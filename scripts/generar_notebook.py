"""
Genera notebooks/fase2_pipeline_ml_mlflow.ipynb.

Por qué existe: el notebook y el paquete `adbd/` tienen que decir exactamente lo
mismo. Mantener dos copias del mismo código a mano es una fábrica de bugs — uno
se corrige y el otro no. Aquí el notebook se ARMA desde los módulos del
repositorio: las celdas `%%writefile` llevan el archivo tal como está en disco, y
el resto del notebook solo orquesta y narra. Una sola fuente de verdad.

    python -m scripts.generar_notebook
"""
from __future__ import annotations

import json
import os

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESTINO = os.path.join(RAIZ, "notebooks", "fase2_pipeline_ml_mlflow.ipynb")

MODULOS = ["config", "utilidades", "contrato", "bronze", "silver", "gold",
           "transformadores", "features", "modelos", "evaluacion", "seguimiento"]


def md(texto):
    return {"cell_type": "markdown", "metadata": {}, "source": texto.strip("\n").splitlines(True)}


def code(texto):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": texto.strip("\n").splitlines(True)}


def leer(ruta):
    with open(os.path.join(RAIZ, ruta), encoding="utf-8") as f:
        return f.read().rstrip("\n")


def celda_modulo(nombre):
    return code(f"%%writefile adbd/{nombre}.py\n" + leer(f"adbd/{nombre}.py"))


def construir():
    c = []

    c.append(md("""
# Fase 2 · Pipeline batch, ML distribuido y MLflow

**Análisis de Big Data · Magíster en Data Science UDD · Proyecto Integrador**
**Dataset:** Ali_Display_Ad_Click (Alimama / Taobao) · **Entrega:** lunes 21 de septiembre de 2026
**Equipo:** Juan José Torres · Claudio Ballerini · Cristian Vargas · Christian Vásquez

**Qué produce.** Cada cifra del informe sale de una celda de aquí y queda en
`resultados_fase2.json`. Es la misma regla de la Fase 1: *ninguna afirmación existe sin una celda
que la imprima*.

**Cómo está armado.** El notebook **escribe el paquete `adbd/`** antes de usarlo (sección 0c). No es
un adorno: la pauta evalúa modularidad y reproducibilidad, y así el mismo código que corre aquí es
el que queda versionado en el repositorio y el que ejecuta `scripts/correr_fase2.py` sin notebook.
Una sola fuente de verdad, no dos copias que se desincronizan.

---

### Lo que la Fase 1 dejó decidido y aquí no se vuelve a discutir

| Decisión de la Fase 1 | Cómo entra en la Fase 2 |
|---|---|
| Parquet + ZSTD particionado por `fecha_local` como frontera | Bronze se **reutiliza** si ya existe y cumple el contrato |
| `time_stamp` está en UTC, no en hora local (UTC+8) | La corrección se aplica una sola vez, en Bronze |
| El segmento **sin perfil** (5,76%) se conserva con flag, no se filtra | `sin_perfil` es un *feature* |
| `brand`: el cast fabricó 246.330 nulos inexistentes | **Descartada** como predictor; sobrevive como `brand_conocida` |
| `pvalue_level` (54,24%) y `new_user_class_level` (32,49%): "se deciden en Fase 2" | Sección 2, **con una medición**. En el cruce los porcentajes son otros —54,80% y 30,97%— y esa diferencia es el punto |
| W3 con `RANGE ... 1 PRECEDING` para no meter fuga | Se mantiene literal como *feature* del slot |
| W2 necesitó desempate `(time_stamp, adgroup_id, pid)` para ser determinista | El mismo desempate ordena las ventanas de fatiga |
| Spark para producción (ALS y streaming no existen fuera de él) | Todo el pipeline es Spark; DuckDB queda para validación cruzada |

### Las cuatro decisiones nuevas de esta fase

1. **Split temporal con un día de burn-in.** `06-may` no se entrena: existe para que los históricos
   del `07-may` no sean nulos. Train `07..11-may`, validación `12-may`, test `13-may` — y el test se
   toca **una** vez.
2. **Ningún feature puede ver el futuro, y hay un control ejecutable que lo verifica.** La Fase 1 ya
   encontró una fuga de este tipo (`CURRENT ROW` en W3); aquí el pipeline **se detiene** si reaparece.
3. **Submuestreo de negativos con recalibración.** 1 de cada 10 negativos, todos los positivos, y la
   probabilidad se corrige. La evaluación va siempre sobre el conjunto **completo**.
4. **La línea base es un modelo, no una frase.** Predecir siempre el CTR histórico fija el piso: sin
   ese número, un AUC-PR de 0,09 no se puede leer.

**Ejecutar:** kernel **`Python (adbd-fase2)`** → `Kernel → Restart & Run All`. Entorno: Colab
(Java 11) o Jupyter con Java 11/17. Duración medida sobre el dataset completo: **51,6 min** en una
estación de 22 núcleos (ver sección 11 · el entorno importa y se declara).
"""))

    # ------------------------------------------------------------------ 0
    c.append(md("## 0. Setup"))

    c.append(md("""
### 0a. La raíz del proyecto

Jupyter arranca el kernel en el directorio del *notebook* (`notebooks/`), pero **todas** las rutas
del proyecto son relativas a la raíz: `./datos_csv`, `./adbd`, `./mlruns`, `./artefactos_fase2`.
Sin este `chdir` el *notebook* no encuentra los CSV —se va por la rama de descarga y falla con un
`ModuleNotFoundError: kagglehub` que no tiene nada que ver con el problema real— y escribe el
paquete `adbd/` **dentro de** `notebooks/`.

La raíz se busca hacia arriba por una marca que el *notebook* no crea (`requirements.txt`). Si no
aparece ninguna —Colab, Databricks—, no se toca nada: ahí el directorio de trabajo ya es el correcto.
"""))
    c.append(code("""
import os, sys

_MARCAS = ("requirements.txt", "datos_csv")
_d, _raiz = os.path.abspath(os.getcwd()), None
while True:
    if any(os.path.exists(os.path.join(_d, _m)) for _m in _MARCAS):
        _raiz = _d
        break
    _padre = os.path.dirname(_d)
    if _padre == _d:          # se llegó a la raíz del disco sin encontrar marca
        break
    _d = _padre

if _raiz and _raiz != os.getcwd():
    os.chdir(_raiz)
print("directorio de trabajo:", os.getcwd())
print("kernel:", sys.executable)
"""))

    c.append(md("""
Y las dependencias. En **Colab y Databricks CE** el entorno es efímero y falta casi todo: se instala
lo que falte. En un **entorno local** esta celda **no instala nada**: si falta un paquete, casi
siempre es porque el notebook se abrió con el kernel de otro proyecto, y llenar ese kernel de
`pyspark` no lo arregla — la celda se detiene y dice qué kernel está corriendo y cuál corresponde.
"""))
    c.append(code("""
import importlib.util, subprocess

_REQ = {"pyspark": "pyspark==3.5.4", "mlflow": "mlflow==2.17.2", "duckdb": "duckdb",
        "polars": "polars", "pandas": "pandas", "pyarrow": "pyarrow",
        "kagglehub": "kagglehub", "psutil": "psutil", "matplotlib": "matplotlib"}
_faltan = [v for k, v in _REQ.items() if importlib.util.find_spec(k) is None]
_efimero = ("google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ
            or "DATABRICKS_RUNTIME_VERSION" in os.environ)

if not _faltan:
    print("todas las dependencias presentes · no se instala nada")
elif _efimero:
    print("entorno efímero · instalando:", " ".join(_faltan))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *_faltan])
else:
    raise ModuleNotFoundError(
        f"faltan {', '.join(_faltan)} en el kernel {sys.executable}"
        "\\n  Este notebook no instala paquetes en un entorno local."
        "\\n  · Elegí el kernel 'Python (adbd-fase2)' en el selector (arriba a la derecha)"
        " y volvé a correr desde el principio."
        "\\n  · Si de verdad querés usar este intérprete: pip install -r requirements.txt")
"""))

    c.append(md("""
### 0b. Acceso a los datos

Dos vías, en este orden: (a) los CSV ya están en `RUTA_CSV`; (b) descarga del espejo de Kaggle.
**La licencia que este trabajo reconoce es la de Tianchi** (`tianchi.aliyun.com/dataset/56`); el
espejo de Kaggle declara `License(s): unknown` y se usa **solo como medio de acceso**, igual que en
la Fase 1.
"""))
    c.append(code("""
import glob, json, time, shutil
import pandas as pd

RUTA_CSV = os.environ.get("ADBD_RUTA_CSV", "./datos_csv")
ARCHIVOS = ["raw_sample.csv", "ad_feature.csv", "user_profile.csv"]

def _hay(ruta):
    return all(glob.glob(os.path.join(ruta, "**", a), recursive=True) for a in ARCHIVOS)

if not _hay(RUTA_CSV):
    import kagglehub
    RUTA_CSV = kagglehub.dataset_download("pavansanagapati/ad-displayclick-data-on-taobaocom")
    print("descargado en", RUTA_CSV)

os.environ["ADBD_RUTA_CSV"] = RUTA_CSV
for a in ARCHIVOS:
    p = glob.glob(os.path.join(RUTA_CSV, "**", a), recursive=True)[0]
    print(f"{a:<18} {os.path.getsize(p)/1e6:8,.1f} MB   {p}")

# behavior_log.csv NO viene en el espejo. Se verifica, no se supone: es la carta de
# escalamiento que la Fase 1 dejó anotada y que la curva de aprendizaje de la sección 7
# va a decidir si conviene jugar.
print("behavior_log.csv presente:",
      bool(glob.glob(os.path.join(RUTA_CSV, "**", "behavior_log.csv"), recursive=True)))
"""))

    # ------------------------------------------------------------------ 0c
    c.append(md("""
### 0c. El paquete `adbd`

Las celdas que siguen **escriben el repositorio**. Cada módulo tiene una responsabilidad y se puede
correr aislado; eso es lo que hace posible `scripts/correr_fase2.py`, que reproduce toda la fase sin
abrir un notebook.

| Módulo | Responsabilidad |
|---|---|
| `config` | constantes y supuestos declarados: split, umbrales, tarifa, semilla |
| `utilidades` | cronómetro por etapa, tamaños en disco, exportación a JSON |
| `contrato` | esquema declarado; **detiene** la carga si falta una columna requerida |
| `bronze` | CSV → Parquet+ZSTD por `fecha_local`, corrección UTC→UTC+8 |
| `silver` | joins, perfilamiento de calidad y decisiones de imputación **medidas** |
| `gold` | *features* con control de fuga, split temporal, registro de variables |
| `transformadores` | `CodificadorCentinela` y `CorrectorPrior` (Transformers propios) |
| `features` | ensamblado del `Pipeline` de MLlib y lectura de importancias |
| `modelos` | submuestreo, catálogo de experimentos, `CrossValidator`, ALS |
| `evaluacion` | AUC-PR, *lift* por decil, calibración, métricas de *ranking* |
| `seguimiento` | MLflow: qué se registra y la tabla comparativa |
"""))
    c.append(code("!mkdir -p adbd scripts tests notebooks artefactos_fase2"))
    for m in MODULOS:
        c.append(celda_modulo(m))
    c.append(code("%%writefile adbd/__init__.py\n" + leer("adbd/__init__.py")))
    c.append(code("""
import importlib, sys
sys.path.insert(0, os.path.abspath("."))
import adbd
importlib.reload(adbd)
from adbd import (bronze, config, evaluacion, features, gold as gold_mod,
                  modelos, seguimiento, silver as silver_mod, utilidades)
from adbd.transformadores import CorrectorPrior
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

config.RUTA_CSV = RUTA_CSV
RES, crono = {}, utilidades.Crono()
print("paquete adbd", adbd.__version__, "listo")
"""))

    c.append(code("""
spark = utilidades.crear_sesion("ADBD-Fase2")
RES["entorno"] = utilidades.huella_entorno(spark)
RES["entorno"]
"""))

    # ------------------------------------------------------------------ 1
    c.append(md("""
## 1. Bronze · la capa que ya existe

La Fase 1 dejó 26.557.961 impresiones en Parquet+ZSTD particionado por `fecha_local`, 376,2 MB en
8 particiones; la de esta fase agrega `ts_local`, `hora_local` y `franja` y por eso pesa **435,0 MB**.
**No se reescribe**: se verifica que cumpla el contrato y se reutiliza. Que el
contrato se **verifique** en vez de suponerse no es ceremonia — un Parquet de una corrida anterior
con el esquema incompleto sobrevive en disco y el pipeline correría igual, fallando recién en la
consulta que usa la columna que falta.

`rehacer=True` reconstruye todo desde los CSV: es la reproducibilidad de extremo a extremo que la
Fase 3 va a exigir, disponible desde ahora.
"""))
    c.append(code("""
with crono.medir("bronze"):
    RES["bronze"] = bronze.construir(spark, rehacer=False)

imp, ads, usr = bronze.leer(spark)
RES["filas_bronze"] = imp.count()
RES["n_avisos"], RES["n_perfiles"] = ads.count(), usr.count()
print(f"impresiones {RES['filas_bronze']:,} · avisos {RES['n_avisos']:,} · perfiles {RES['n_perfiles']:,}")
"""))

    # ------------------------------------------------------------------ 2
    c.append(md("""
## 2. Silver · calidad, y las dos decisiones que la Fase 1 dejó pendientes

La Fase 1 midió `pvalue_level` **54,24%** de nulos y `new_user_class_level` **32,49%**, y escribió
que se decidían acá. La respuesta refleja sería imputar con la moda. Antes de decidir se mide algo
concreto: **¿el grupo sin dato clickea distinto del grupo con dato?**

Si la respuesta es sí, "no sé" es información sobre el usuario y sustituirla por la moda la
destruye. Es la misma lógica que en la Fase 1 salvó al segmento sin perfil: filtrarlo era la
decisión obvia y habría borrado el segundo grupo más grande, que además clickea **por encima** del
promedio (5,33% contra 5,14%).

Y la completitud que importa es **la del cruce**, no la de la tabla de perfiles: dentro de
`user_profile.csv` varias columnas tienen 0% de nulos, pero en el JOIN falta el 5,76%. Por la misma
razón los porcentajes de la Fase 1 no se repiten aquí: `pvalue_level` tenía 54,24% de nulos *dentro
de la tabla* y tiene **54,80%** en el cruce; `new_user_class_level`, 32,49% contra **30,97%**.

**Qué decide "informativa", y por qué no es el tamaño del efecto.** La primera versión de esta
regla usaba un umbral fijo sobre la razón de CTR (5%). Sobre 26,6M de filas ese umbral está mal
calibrado en las dos direcciones: el grupo sin perfil tiene 5,335% de CTR contra 5,132% del grupo
con perfil —razón 1,040, *por debajo* del umbral— y sin embargo la diferencia tiene **z = 11,0**.
Con el umbral viejo las ocho columnas salían marcadas *"ausencia no informativa"*, contradiciendo
el hallazgo de la Fase 1. Lo que la regla necesita saber es si la diferencia **existe**, y eso es
una prueba de dos proporciones; el tamaño del efecto se reporta aparte para que el lector juzgue si
además importa.
"""))
    c.append(code("""
with crono.medir("perfilamiento de calidad"):
    perfil = silver_mod.perfilar_calidad(spark)
    decisiones = silver_mod.decidir_imputacion(perfil)
RES["calidad"] = decisiones.to_dict("records")
decisiones
"""))
    c.append(md("""
**Control heredado: la cardinalidad antes de leer cualquier tasa.** Un JOIN mal especificado sobre
`adgroup_id` infla las filas y el CTR resultante *sigue pareciendo razonable*. El `assert` detiene
el pipeline si algún join mueve el universo.
"""))
    c.append(code("""
with crono.medir("silver"):
    card = silver_mod.validar_cardinalidad(spark, RES["filas_bronze"])
    silver = silver_mod.construir(spark, decisiones)
    RES["filas_silver"] = silver.count()
RES["cardinalidad"] = card.to_dict("records")
print(f"Silver: {RES['filas_silver']:,} filas")
card
"""))

    # ------------------------------------------------------------------ 3
    c.append(md("""
## 3. Gold · *features*, y el riesgo que define esta fase

Aquí se gana o se pierde la Fase 2, porque aquí vive la **fuga de información**. La Fase 1 ya
encontró una: `CURRENT ROW` en lugar de `1 PRECEDING` en W3 metía el resultado a predecir dentro
del *feature*, y el split temporal seguía viéndose impecable. Un AUC inflado no falla, miente.

**Regla única:** ningún *feature* puede depender de una fila cuyo `clk` el modelo todavía no habría
visto en el instante de la impresión.

De ahí salen las tres familias de *features* derivados del objetivo:

| Familia | Cómo se difiere | Qué captura |
|---|---|---|
| CTR histórico por entidad (`cate_id`, `campaign_id`, `customer`, `pid`) | ventana expansiva con **retardo de un día**: para el día *d* solo suma días *< d* | calidad del inventario |
| CTR y fatiga del usuario | ventana expansiva **dentro del flujo**, `rowsBetween(unboundedPreceding, -1)`, con el desempate total de W2 | el hallazgo más accionable de la Fase 1: el CTR cae de 7,01% en la primera impresión a 4,30% de la vigésima |
| CTR móvil del slot | `RANGE BETWEEN 6 PRECEDING AND 1 PRECEDING`, literal de W3 | tendencia reciente de la posición |

Todos suavizados por m-estimación, `(clicks_previos + m·p₀) / (impresiones_previas + m)`, con el
prior `p₀` **también diferido**. Una entidad nueva recibe el prior, no un `NaN` ni un cero engañoso.

**El costo de la regla es un día.** El `06-may` no se entrena: existe para que el `07-may` tenga
historia. Se paga y se declara.
"""))
    c.append(code("""
with crono.medir("gold"):
    gold = gold_mod.construir(spark, silver)
    RES["filas_gold"] = gold.count()

resumen = gold_mod.resumen_split(gold)
RES["split"] = resumen.to_dict("records")
print("El CTR por día tiene que ser estable: si el día de test tuviera un CTR muy distinto,")
print("comparar métricas entre validación y test no significaría nada.\\n")
resumen
"""))
    c.append(md("""
### 3.1 El control de fuga, ejecutable

Tres pruebas. Si alguna falla el notebook **se detiene**: publicar métricas de un modelo con fuga
es peor que no tener modelo.

1. Ninguna correlación `|r| > 0,95` entre un *feature* numérico y el objetivo.
2. El primer día entrenable no tiene históricos nulos — el burn-in cumplió su función.
3. Los CTR históricos caen dentro de `(0, 1]`.
"""))
    c.append(code("""
with crono.medir("control de fuga"):
    fuga = gold_mod.verificar_fuga(gold, spark)
RES["fuga"] = fuga.to_dict("records")
assert (fuga.veredicto == "OK").all(), "control de fuga NO superado; no se sigue"
fuga
"""))
    c.append(md("""
### 3.2 Registro de variables versionado (`gold_ads_v1`)

El artefacto que la Fase 1 prometió: lo aceptado **y lo descartado con su motivo**. Lo segundo es lo
que evita que en dos semanas alguien reincorpore `brand` sin saber que sus nulos los fabricó la
línea de carga.
"""))
    c.append(code("""
registro = gold_mod.registro_variables()
RES["registro_variables"] = registro.to_dict("records")
print(f"{(registro.estado=='aceptada').sum()} variables aceptadas · "
      f"{(registro.estado=='DESCARTADA').sum()} descartadas con motivo\\n")
registro[registro.estado == "DESCARTADA"][["variable", "motivo"]]
"""))
    c.append(code("""
train, val, test = gold_mod.particionar(gold)
for d in (train, val, test):
    d.cache()
RES["n_train"], RES["n_val"], RES["n_test"] = train.count(), val.count(), test.count()
print(f"train {RES['n_train']:,} · val {RES['n_val']:,} · test {RES['n_test']:,}")
"""))

    # ------------------------------------------------------------------ 4
    c.append(md("""
## 4. Desbalance: submuestreo de negativos y su precio

Con CTR 5,19% en la ventana de entrenamiento (≈1:18), entrenar sobre sus **16.744.897 filas** con
`CrossValidator` de 3 pliegues y las 12 combinaciones que suman las cuatro grillas son **36 ajustes
completos**. Se conserva **1 de cada 10 negativos y todos los positivos**, y el entrenamiento baja a
**2.458.575 filas**.

El atajo tiene dos costos y los dos se pagan explícitamente:

1. **La probabilidad queda inflada.** El modelo aprende sobre una prevalencia inventada. La
   corrección es exacta: `p = p_s·r / (p_s·r + 1 − p_s)`, implementada como un Transformer
   (`CorrectorPrior`) y probada contra casos cerrados en `tests/test_humo.py`.
2. **La evaluación no puede hacerse sobre la muestra.** AUC-PR depende de la prevalencia. La
   sección 8 lo demuestra con números en vez de afirmarlo.

Se usa la tasa **efectiva** —la fracción realmente conservada— y no la nominal: `sample` es de
Bernoulli por fila y con la nominal la corrección queda sesgada en el tercer decimal.
"""))
    c.append(code("""
with crono.medir("submuestreo de negativos"):
    train_ds, info_ds = modelos.submuestrear_negativos(train)
    train_ds = train_ds.cache(); train_ds.count()
RES["submuestreo"] = info_ds
r_efectiva = info_ds["tasa_efectiva"]
corrector = CorrectorPrior(inputCol="probability", outputCol="p_calibrada",
                           tasaNegativos=r_efectiva)
pd.DataFrame([info_ds]).T.rename(columns={0: "valor"})
"""))

    # ------------------------------------------------------------------ 5
    c.append(md("""
## 5. MLflow y la línea base

`file:./mlruns` — backend de archivos, que es lo que funciona en Colab gratuito sin servidor ni base
de datos. Se entrega comprimido con el repositorio para que el *tracking* sea auditable.

En cada run se registran los parámetros del **modelo** y los del **pipeline de datos** (tasa de
submuestreo, `m` de suavizado, fechas del split). Sin los segundos, dos runs con el mismo AUC son
indistinguibles y no se sabe cuál reproducir.

**La línea base primero.** Predecir siempre el CTR histórico no es relleno: fija el piso. Con
prevalencia 5,14%, un clasificador que dice siempre "no click" tiene 94,86% de *accuracy* y cero
valor, y su AUC-PR es exactamente la prevalencia. Sin este número no se puede leer ningún otro.
"""))
    c.append(code("""
seguimiento.iniciar()
params_datos = seguimiento.parametros_datos(info_ds)

p0, base_test = modelos.linea_base_prior(train, test)
m_base = evaluacion.metricas_binarias(base_test)
RES["linea_base"] = {"p0": p0, **m_base}
seguimiento.registrar_run(
    "baseline_prior", "¿Cuál es el piso real? Predecir siempre el CTR histórico.",
    {**params_datos, "modelo": "constante", "p0": round(p0, 6)},
    {f"test_{k}": v for k, v in m_base.items() if isinstance(v, (int, float))})
print(f"línea base · AUC-PR {m_base['auc_pr']:.5f} = prevalencia {m_base['ctr_observado']:.5f} "
      f"· AUC-ROC {m_base['auc_roc']:.3f} (0,5 por construcción)")
"""))

    # ------------------------------------------------------------------ 6
    c.append(md("""
## 6. Los experimentos

Cuatro, y **cada uno responde una pregunta**. Un barrido de modelos sin pregunta produce una tabla
que no se puede defender ante el panel.

| Experimento | Pregunta |
|---|---|
| `lr_contexto` | ¿Cuánto se predice **sin historia**, solo con perfil y contexto? Aísla el aporte real de los *features* derivados del objetivo. |
| `lr_completo` | ¿Cuánto agregan los CTR históricos diferidos y la fatiga? |
| `rf_completo` | ¿Gana un *ensemble* de árboles sin interacciones explícitas? |
| `gbt_completo` | ¿Compensa el *boosting* su costo de cómputo en AUC-PR? |

**Sobre el `CrossValidator` y el split temporal.** El k-fold aleatorio parte la ventana de
entrenamiento sin respetar el orden, lo que normalmente sería una fuga. Aquí no lo es, porque la
protección temporal vive en los **features** —`gold.py` los construye diferidos— y no en el corte:
una fila del 10-may no contiene nada del 11-may aunque caiga en el mismo pliegue. La afirmación no
se deja en palabras: se mide la **brecha de optimismo** contra el día de validación retenido.

**Y esa brecha hay que medirla bien.** El `CrossValidator` calcula su AUC-PR sobre pliegues ya
submuestreados (prevalencia 35,4%) y la validación va sobre el día completo (5,0%). Restar esos dos
números da una diferencia enorme que **no mide optimismo sino prevalencia**. La comparación válida
es contra una validación submuestreada a la misma tasa, con el contraste en AUC-ROC —invariante al
submuestreo— como control cruzado.

**Presupuesto de cómputo, declarado.** La *búsqueda* corre sobre el 20% del train ya submuestreado;
el modelo **ganador** se reajusta sobre el train submuestreado completo.
"""))
    c.append(code("""
numericas_sin_historia = [c for c in gold_mod.NUMERICAS
                          if not c.startswith(("ctr_hist_", "log_imp_hist_", "ctr_usuario",
                                               "log_imp_previas", "ctr_movil"))]
catalogo = modelos.catalogo_experimentos(numericas_sin_historia)
resultados, curvas, modelos_ajustados = {}, {}, {}

for nombre, spec in catalogo.items():
    print(f"\\n===== {nombre} · {spec['pregunta']}")
    with crono.medir(f"experimento {nombre}"):
        modelo, info_cv = modelos.ajustar_con_cv(train_ds, spec)

    pv = corrector.transform(modelo.transform(val))
    pt = corrector.transform(modelo.transform(test))
    m_val, m_test = evaluacion.metricas_binarias(pv), evaluacion.metricas_binarias(pt)
    deciles = evaluacion.tabla_deciles(pt)
    curvas[nombre] = evaluacion.curva_pr(pt)
    lift_d1 = float(deciles.iloc[0]["lift"])

    val_ds, _ = modelos.submuestrear_negativos(val, r_efectiva)
    m_val_ds = evaluacion.metricas_binarias(corrector.transform(modelo.transform(val_ds)))
    brecha = info_cv["cv_mejor"] - m_val_ds["auc_pr"]
    brecha_roc = m_val_ds["auc_roc"] - m_val["auc_roc"]

    metricas = ({f"val_{k}": v for k, v in m_val.items() if isinstance(v, (int, float))}
                | {f"test_{k}": v for k, v in m_test.items() if isinstance(v, (int, float))}
                | {"test_lift_d1": lift_d1, "cv_auc_pr": info_cv["cv_mejor"],
                   "val_ds_auc_pr": m_val_ds["auc_pr"],
                   "brecha_optimismo_cv_val": brecha,
                   "brecha_auc_roc_muestra_vs_completo": brecha_roc,
                   "t_busqueda_s": info_cv["t_busqueda_s"],
                   "t_ajuste_final_s": info_cv["t_ajuste_final_s"]})
    run_id = seguimiento.registrar_run(
        nombre, spec["pregunta"],
        {**params_datos, **info_cv["mejores_parametros"],
         "estimador": type(spec["estimador"]).__name__,
         "escalado": spec.get("escalar", False),
         "n_combinaciones": info_cv["n_combinaciones"], "folds": info_cv["folds"],
         "fraccion_busqueda": info_cv["fraccion_busqueda"]},
        metricas, tablas={"deciles": deciles, "curva_pr": curvas[nombre]})

    imp = features.importancias(modelo, numericas=spec.get("numericas"))
    modelos_ajustados[nombre] = modelo
    resultados[nombre] = {"run_id": run_id, "cv": info_cv, "val": m_val, "test": m_test,
                          "val_submuestreada": m_val_ds, "lift_d1": lift_d1,
                          "brecha_optimismo": brecha, "brecha_auc_roc": brecha_roc,
                          "importancias": imp.to_dict("records"),
                          "deciles": deciles.to_dict("records")}
    print(f"  AUC-PR test {m_test['auc_pr']:.5f} · AUC-ROC {m_test['auc_roc']:.5f} "
          f"· lift D1 {lift_d1:.2f}x · brecha CV-val {brecha:+.5f}")

RES["experimentos"] = resultados
RES["mejor_experimento"] = max(resultados, key=lambda k: resultados[k]["test"]["auc_pr"])
print("\\nmejor por AUC-PR en test:", RES["mejor_experimento"])
"""))

    c.append(md("""
### 6.1 Qué mueve la predicción

Los coeficientes y las importancias sirven para dos cosas distintas y las dos importan en la
defensa: explicar el modelo, y **detectar fuga que las métricas no delatan**. Una variable que
concentra casi toda la importancia suele ser una fuga, no un hallazgo.

Los nombres se leen de `categorySizes` del `OneHotEncoderModel` ajustado y no se deducen a mano:
deducirlos desalinea los nombres por una posición y hace que el informe atribuya un coeficiente a la
variable equivocada — un error que no se ve porque no falla.
"""))
    c.append(code("""
mejor = RES["mejor_experimento"]
imp = pd.DataFrame(resultados[mejor]["importancias"])
print(f"{mejor} · top 15 variables\\n")
imp.head(15)
"""))

    c.append(md("""
### 6.2 El contraste: ¿cuánto cuesta el atajo?

El mismo modelo sobre el train **completo**, sin submuestrear, con `weightCol`. Sin este
experimento, "submuestreamos por costo" es una afirmación sin número detrás.

Un detalle que el experimento mismo enseña: `weightCol` **tampoco** deja la probabilidad calibrada.
Pesar los positivos por *w* multiplica sus *odds* por *w*, igual que el submuestreo las multiplica
por `1/r`. Es la misma corrección con `r = 1/w`, y omitirla sería el error que este experimento
existe para desmentir.
"""))
    c.append(code("""
spec = catalogo["lr_completo"]
with crono.medir("experimento lr_pesos_completo"):
    train_w, info_w = modelos.agregar_pesos(train)
    est = spec["estimador"].copy(); est.setWeightCol("peso")
    pipe_w = features.pipeline(est, numericas=spec.get("numericas"), escalar=spec.get("escalar"))
    t0 = time.perf_counter(); modelo_w = pipe_w.fit(train_w); t_w = round(time.perf_counter() - t0, 1)

corrector_w = CorrectorPrior(inputCol="probability", outputCol="p_calibrada",
                             tasaNegativos=1.0 / info_w["peso_positivo"])
pt_w = corrector_w.transform(modelo_w.transform(test))
m_test_w, dec_w = evaluacion.metricas_binarias(pt_w), evaluacion.tabla_deciles(pt_w)

RES["contraste_pesos"] = {**info_w, "t_ajuste_final_s": t_w, "test": m_test_w,
                          "lift_d1": float(dec_w.iloc[0]["lift"]),
                          "delta_auc_pr_vs_submuestreo":
                              round(m_test_w["auc_pr"] - resultados["lr_completo"]["test"]["auc_pr"], 6),
                          "razon_tiempo": round(t_w / max(resultados["lr_completo"]["cv"]["t_ajuste_final_s"], 1e-9), 2)}
seguimiento.registrar_run(
    "lr_pesos_completo",
    "¿Cuánto AUC-PR cuesta el submuestreo frente a entrenar con el dataset completo?",
    {**params_datos, "estimador": "LogisticRegression", "weightCol": "peso",
     "submuestreo": "no", "peso_positivo": info_w["peso_positivo"]},
    {f"test_{k}": v for k, v in m_test_w.items() if isinstance(v, (int, float))}
    | {"t_ajuste_final_s": t_w, "test_lift_d1": float(dec_w.iloc[0]["lift"])},
    tablas={"deciles": dec_w})

pd.DataFrame([
    {"variante": "submuestreo 1:10 + recalibración",
     "filas": info_ds["filas_entrenamiento"],
     "t_ajuste_s": resultados["lr_completo"]["cv"]["t_ajuste_final_s"],
     "auc_pr_test": resultados["lr_completo"]["test"]["auc_pr"],
     "logloss": resultados["lr_completo"]["test"]["logloss"]},
    {"variante": "train completo + weightCol",
     "filas": info_w["filas"], "t_ajuste_s": t_w,
     "auc_pr_test": m_test_w["auc_pr"], "logloss": m_test_w["logloss"]},
])
"""))

    # ------------------------------------------------------------------ 7
    c.append(md("""
## 7. Escalamiento: ¿más datos o más modelo?

La Fase 1 dejó una pregunta abierta con nombre propio: **¿conviene incorporar `behavior_log`**
(~704M de registros, 26× el volumen) que el espejo de Kaggle no trae? La respuesta no es una
intuición: se ajusta el mejor modelo sobre fracciones crecientes del entrenamiento y se mira si la
curva de aprendizaje **ya está plana**.

Si lo está, 26× más volumen compra tiempo de cómputo y no precisión, y eso es una decisión de
escalamiento defendible con un número. El tiempo de ajuste en el eje derecho es la otra mitad del
argumento: dice si el costo crece lineal con las filas o peor.

**Cuidado con lo que la curva NO dice.** Si sale plana, lo está *para este espacio de features y este
modelo*. `behavior_log` no traería solo más filas de lo mismo: traería el comportamiento de
navegación, que son *features* que hoy no existen. La conclusión honesta es «más filas de las mismas
variables no compran precisión», no «más datos nunca sirven».
"""))
    c.append(code("""
with crono.medir("curva de aprendizaje"):
    d_curva = modelos.curva_aprendizaje(
        train_ds, catalogo[RES["mejor_experimento"]],
        fracciones=(0.1, 0.25, 0.5, 1.0), evaluar=val, corrector=corrector,
        metrica_fn=evaluacion.metricas_binarias)
RES["curva_aprendizaje"] = d_curva.to_dict("records")
evaluacion.graficar_curva_aprendizaje(d_curva, "artefactos_fase2/curva_aprendizaje.png")
d_curva
"""))

    # ------------------------------------------------------------------ 8
    c.append(md("""
## 8. Por qué la evaluación va sobre el conjunto completo

La demostración, con números. El mismo modelo, evaluado sobre el día de test completo y sobre el
mismo día submuestreado a la tasa de entrenamiento:

- **AUC-ROC apenas se mueve**: es puro orden, y el submuestreo de negativos no reordena nada.
- **AUC-PR se dispara**: depende de la prevalencia por definición.

Reportar el segundo número sería reportar un modelo que no existe.
"""))
    c.append(code("""
m = modelos_ajustados[RES["mejor_experimento"]]
test_ds, _ = modelos.submuestrear_negativos(test, r_efectiva)
comp = evaluacion.comparar_submuestreo(corrector.transform(m.transform(test)),
                                       corrector.transform(m.transform(test_ds)))
RES["efecto_submuestreo_en_la_evaluacion"] = comp.to_dict("records")
comp
"""))
    c.append(md("""
### 8.1 La traducción a negocio: *lift* por decil

El panel de la defensa no compra un AUC. En una plataforma de display no se decide "click o no
click": se **ordena inventario**. Lo que importa es cuánto mejor rinde el 10% de impresiones que el
modelo pone arriba (D1) comparado con servir al azar, y si la probabilidad está lo bastante
calibrada como para poner un umbral de negocio sobre ella.
"""))
    c.append(code("""
deciles = pd.DataFrame(resultados[RES["mejor_experimento"]]["deciles"])
evaluacion.graficar_pr(curvas, RES["linea_base"]["ctr_observado"], "artefactos_fase2/curva_pr.png")
evaluacion.graficar_calibracion(deciles, "artefactos_fase2/calibracion.png")
print(f"D1 rinde {deciles.iloc[0]['lift']:.2f}x el CTR global · "
      f"D10 rinde {deciles.iloc[-1]['lift']:.2f}x\\n")
deciles
"""))

    # ------------------------------------------------------------------ 9
    c.append(md("""
## 9. ALS · recomendación con *feedback* implícito

El segundo eje de modelamiento distribuido que la Fase 1 anticipó: matriz **usuario × categoría**
con los clicks como confianza. Solo entran clicks, porque una impresión sin click no es una señal
negativa sino ausencia de evidencia — que es exactamente el supuesto que `implicitPrefs=True`
modela.

**Contra la popularidad, o no significa nada.** En *feedback* implícito la línea base de
popularidad es sorprendentemente difícil de batir, y un MAP@10 sin ese contraste es un número
suelto. Los usuarios *fríos* —sin historia en train— se cuentan y se reportan por separado:
esconderlos infla todas las métricas, porque ALS no puede recomendarles nada personalizado y la
cobertura es parte del resultado, no una nota al pie.
"""))
    c.append(code("""
with crono.medir("ALS implícito"):
    m_train = modelos.matriz_implicita(train).cache()
    m_test_mat = modelos.matriz_implicita(test).cache()
    RES["als_ids"] = modelos.verificar_ids_als(m_train, m_test_mat)
    als_modelo, t_als = modelos.entrenar_als(m_train)

    recs = (als_modelo.recommendForAllUsers(config.TOP_K)
            .select(F.col("userid").cast("long").alias("userid"),
                    F.col("recommendations.cate_id").alias("recomendadas")))
    reales = (m_test_mat.groupBy("userid")
              .agg(F.collect_set(F.col("cate_id").cast("int")).alias("reales")))
    m_als = evaluacion.metricas_ranking(recs, reales)

    top = modelos.recomendaciones_populares(m_train)
    pop = reales.select("userid").withColumn("recomendadas",
                                             F.array(*[F.lit(int(x)) for x in top]))
    m_pop = evaluacion.metricas_ranking(pop, reales)

RES["als"] = {"als": m_als, "popularidad": m_pop, "t_ajuste_s": t_als,
              "rank": config.ALS_RANK, "alpha": config.ALS_ALPHA}
seguimiento.registrar_run(
    "als_implicito",
    "¿Un modelo de recomendación bate a la popularidad en exposición por categoría?",
    {**params_datos, "modelo": "ALS", "rank": config.ALS_RANK, "regParam": config.ALS_REG,
     "alpha": config.ALS_ALPHA, "implicitPrefs": True},
    {f"test_{k}": v for k, v in m_als.items() if isinstance(v, (int, float))}
    | {f"pop_{k}": v for k, v in m_pop.items() if isinstance(v, (int, float))}
    | {"t_ajuste_final_s": t_als})

pd.DataFrame([{"modelo": "ALS implícito", **m_als}, {"modelo": "popularidad", **m_pop}])
"""))

    # ------------------------------------------------------------------ 10
    c.append(md("""
## 10. Comparación de experimentos

La tabla sale **de MLflow**, no de variables en memoria. Si la tabla del informe se lee desde el
*tracking*, el *tracking* es la fuente de verdad y no una decoración que se llenó por cumplir.
"""))
    c.append(code("""
comparativa = seguimiento.tabla_comparativa()
RES["comparativa_mlflow"] = comparativa.to_dict("records")
comparativa
"""))

    # ------------------------------------------------------------------ 11
    c.append(md("""
## 11. Tiempo y costo (FinOps)

Misma tarifa que la Fase 1 —**US$ 0,27/hora**, e2-standard-8 *on-demand*— para que las dos fases se
valoricen en la misma unidad. El costo monetario efectivo sigue siendo **US$ 0**.

**Pero los tiempos de las dos fases NO son comparables entre sí, y hay que decirlo antes de la
tabla.** El *benchmark* de la Fase 1 corrió en 2 núcleos con 13,6 GB de RAM; esta fase corrió en la
máquina que imprime la sección 0 (22 núcleos, 102,6 GB). Comparar 51,6 min de aquí contra 26,0 min
de allá no mide progreso ni regresión: mide dos máquinas distintas. Lo que sí es comparable es el
**reparto interno** de cada fase —dónde se va el tiempo— y el costo por hora, que es el mismo.
"""))
    c.append(code("""
tiempos = crono.tabla()
RES["tiempos"] = crono.marcas
RES["t_total_s"] = crono.total()
RES["usd_referencial"] = utilidades.usd(crono.total())
RES["extrapolacion_behavior_log"] = {
    "factor": round(config.FACTOR_BEHAVIOR, 1),
    "segundos_lineales": round(crono.total() * config.FACTOR_BEHAVIOR, 1),
    "usd_lineales": utilidades.usd(crono.total() * config.FACTOR_BEHAVIOR),
    "nota": ("la extrapolación es LINEAL y por eso es un piso: 26x el volumen no cabe en RAM "
             "y un motor de un solo nodo se degrada de forma no lineal"),
}
print(f"TOTAL {RES['t_total_s']:,} s ({RES['t_total_s']/60:.1f} min) · "
      f"US$ {RES['usd_referencial']} de cómputo equivalente\\n")
tiempos
"""))

    # ------------------------------------------------------------------ 12
    c.append(md("""
## 12. Exportación

Cada cifra del informe, con su clave. `resultados_fase2.json` es lo que hace verificable la regla
del equipo: *ninguna afirmación se escribe sin una celda que la imprima*.
"""))
    c.append(code("""
RES["notebook"] = "fase2_pipeline_ml_mlflow.ipynb"
utilidades.exportar(RES, "resultados_fase2.json")
shutil.make_archive("mlruns_fase2", "zip", config.RUTA_MLRUNS)
print("mlruns_fase2.zip escrito (tracking completo para auditar)\\n")

for k in ("filas_bronze", "filas_silver", "filas_gold", "n_train", "n_val", "n_test",
          "mejor_experimento", "t_total_s", "usd_referencial"):
    print(f"  {k:<22} = {RES.get(k)}")
print()
for n, r in RES["experimentos"].items():
    print(f"  {n:<14} AUC-PR {r['test']['auc_pr']:.5f} · AUC-ROC {r['test']['auc_roc']:.5f} "
          f"· lift D1 {r['lift_d1']:.2f}x")
"""))

    # ------------------------------------------------------------------ 13
    c.append(md("""
## 13. Declaración de uso de IA generativa

**Sí se usaron asistentes de IA.** Herramienta: **Claude (Anthropic), en Claude Code**.

| Dónde | Uso | Cómo se validó |
|---|---|---|
| Diseño del control de fuga | Primera versión de las ventanas diferidas y del suavizado por m-estimación | El equipo agregó `verificar_fuga()` como `assert` que detiene el pipeline, y el día de burn-in, que el borrador no contemplaba |
| Codificación de *features* | Ensamblado del `Pipeline` de MLlib | Se corrigió la lectura de nombres: el borrador deducía el ancho de cada bloque one-hot a mano y desalineaba los coeficientes por una posición |
| Recalibración tras submuestreo | Fórmula y su implementación como Transformer | Probada contra casos cerrados en `tests/test_humo.py`, no contra sí misma; se detectó que `weightCol` necesita la **misma** corrección y el borrador no la aplicaba |
| Catálogo de experimentos | Grillas y estructura de los runs de MLflow | El equipo exigió que cada experimento tuviera una pregunta declarada; los que no la tenían se eliminaron |
| Regla de calidad de datos | Umbrales y decisión de imputación | Reemplazada por una prueba de dos proporciones tras detectar que el umbral fijo contradecía el hallazgo de la Fase 1 sobre el segmento sin perfil |
| Redacción | Estructura y primer borrador del informe | Cada cifra sale de una celda y se exporta a `resultados_fase2.json` |

**Cuatro correcciones del equipo sobre lo que produjo el asistente.** Ninguna la habría detectado
una prueba de "¿corre?":

1. **La "brecha de optimismo" comparaba peras con manzanas.** El borrador restaba el AUC-PR del
   `CrossValidator` —calculado sobre pliegues submuestreados al ~31% de positivos— del AUC-PR de la
   validación completa al ~5%. La diferencia resultante era enorme y no medía optimismo sino
   **prevalencia**. Se corrigió midiendo contra una validación submuestreada a la misma tasa, con el
   contraste en AUC-ROC como control cruzado.
2. **`weightCol` también descalibra, y el borrador no lo corregía.** Quedaba con un *LogLoss* 3,6×
   peor que la variante submuestreada, y la conclusión "el submuestreo calibra mejor" habría sido
   falsa: el problema era la corrección faltante, no el método.
3. **Faltaba el día de burn-in.** Sin él, los *features* históricos del primer día de entrenamiento
   son nulos y la alternativa del borrador —rellenarlos con el prior global— usa el CTR de todo el
   período, incluido el futuro. Fuga silenciosa.
4. **La línea base no estaba.** El primer borrador comparaba modelos entre sí. Sin el piso
   —AUC-PR = prevalencia— ninguna de esas cifras se puede leer, y menos defender.
5. **La regla de imputación usaba un umbral fijo donde hacía falta una prueba.** "Informativa si el
   CTR difiere más de un 5%" declaraba *no informativas* las ocho columnas de perfil, cuando la
   diferencia del grupo sin perfil tiene **z = 11,0** sobre 1,5M de observaciones. La detectó el
   equipo al ver que el veredicto contradecía un hallazgo ya publicado en la Fase 1 — no un error de
   ejecución, sino un resultado que no encajaba con lo que ya se sabía.

**Las tres preguntas de la clase.** *¿Responde la pregunta real?* Sí: la pregunta es ordenar
inventario, por eso la métrica de cabecera es AUC-PR y la de negocio es el *lift* del decil
superior, no *accuracy*. *¿Se validó contra números conocidos?* Sí: el CTR reconstruido, la línea
base igual a la prevalencia por construcción, la cardinalidad de los joins con `delta = 0` y la
recalibración probada contra casos cerrados. *¿Se revisó el plan de ejecución?* La lectura de planes
fue el núcleo de la Fase 1; en la Fase 2 su equivalente es el control de fuga y la curva de
aprendizaje, que son los que dicen si el número es defendible.

---

**Anexo · Reproducibilidad.** `fase2_pipeline_ml_mlflow.ipynb` (Kernel → Restart & Run All) ·
`resultados_fase2.json` · `mlruns_fase2.zip` (tracking completo) · `artefactos_fase2/` (curvas,
deciles, importancias) · `pytest tests/ -q` verifica el pipeline completo sobre datos sintéticos en
~75 s sin descargar 1,1 GB. Lo único que varía entre corridas son los tiempos.
"""))

    return {
        "cells": c,
        "metadata": {
            "kernelspec": {"display_name": "Python (adbd-fase2)", "language": "python", "name": "adbd-fase2"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    nb = construir()
    os.makedirs(os.path.dirname(DESTINO), exist_ok=True)
    with open(DESTINO, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    n_code = sum(1 for x in nb["cells"] if x["cell_type"] == "code")
    print(f"{DESTINO} · {len(nb['cells'])} celdas ({n_code} de código)")
