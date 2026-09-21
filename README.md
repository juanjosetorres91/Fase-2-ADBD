# Proyecto Integrador · Análisis de Big Data · Magíster en Data Science UDD

**Equipo:** Juan José Torres · Claudio Ballerini · Cristian Vargas · Christian Vásquez
**Dataset:** Ali_Display_Ad_Click (Alimama / Taobao) — 26.557.961 impresiones, 8 días
**Fase actual:** 2 · pipeline batch + ML distribuido + MLflow — **corrida completa sobre los
26.557.961 registros reales: 37/37 celdas, 0 errores, 51,6 min**

---

## La pregunta

> ¿Qué impresiones publicitarias van a generar click, y el modelo resultante expone
> sistemáticamente distinta publicidad —o distintos rangos de precio— según el
> género, el rango etario o el nivel socioeconómico de ciudad del usuario?

**Objetivo:** `clk` (binaria). CTR global **5,14%** (≈1:18). La segunda mitad de la
pregunta se responde en la Fase 3; la Fase 2 deja el modelo y los atributos
protegidos listos para medirla.

## Cómo correrlo

```bash
pip install -r requirements.txt

# a) de punta a punta, sin notebook (es la prueba de reproducibilidad)
python -m scripts.correr_fase2 --ruta-csv ./datos_csv --con-pesos --curva-aprendizaje

# b) el notebook con la narrativa y las tablas del informe
jupyter lab notebooks/fase2_pipeline_ml_mlflow.ipynb     # Kernel -> Restart & Run All

# c) verificar que todo funciona en ~75 s, sin descargar 1,1 GB
python -m scripts.generar_datos_sinteticos --salida ./datos_csv_sinteticos
pytest tests/ -q

# d) regenerar el informe con las cifras de la corrida (ninguna se teclea a mano)
node scripts/generar_informe.js resultados_fase2.json informe_fase2_ADBD.docx
```

Si falta alguna clave en `resultados_fase2.json`, el generador del informe la deja
marcada como `⟦clave⟧` resaltada en el documento y la lista por consola: se ve de
inmediato qué falta correr, en vez de quedar un número inventado.

Si los CSV no están en `--ruta-csv`, el pipeline los descarga del espejo de Kaggle
con `kagglehub`. **La licencia que este trabajo reconoce es la de Tianchi**
(<https://tianchi.aliyun.com/dataset/56>); el espejo es solo un medio de acceso.

### Variables de entorno

| Variable | Default | Para qué |
|---|---|---|
| `ADBD_RUTA_CSV` | `./datos_csv` | dónde están los CSV crudos |
| `ADBD_RUTA_PARQUET` | `./datos_parquet` | dónde viven Bronze/Silver/Gold |
| `ADBD_RUTA_MLRUNS` | `./mlruns` | backend de archivos de MLflow |
| `ADBD_DRIVER_MEM` | `8g` | memoria del driver (bajar en máquinas chicas) |
| `ADBD_ARROW` | `true` | poner en `false` si la JVM es JDK 21 |

### Entornos

Spark 3.5 necesita **Java 11 o 17**. `adbd.utilidades` resuelve solo los dos tropiezos de entorno que
no son culpa del pipeline y que fallan con mensajes que no mencionan la causa:

- **`JAVA_GATEWAY_EXITED`** cuando el JDK está dentro del entorno conda y el kernel arrancó sin
  `conda activate` → `preparar_java()` lo busca en `sys.prefix` y lo pone en el PATH.
- **`HADOOP_HOME and hadoop.home.dir are unset`** en Windows → `preparar_hadoop_windows()` busca
  `winutils.exe` y `hadoop.dll` en `vendor/hadoop/bin` o en `HADOOP_HOME`. En Linux, macOS y Colab no
  hace nada.

## Estructura

```
adbd/
  config.py          constantes y supuestos declarados (split, umbrales, tarifa)
  contrato.py        esquema declarado; detiene la carga si falta una columna requerida
  bronze.py          CSV -> Parquet+ZSTD por fecha_local · corrección UTC -> UTC+8
  silver.py          joins, perfilamiento de calidad y decisiones de imputación medidas
  gold.py            features con control de fuga, split temporal, registro de variables
  transformadores.py Transformers propios: CodificadorCentinela y CorrectorPrior
  features.py        ensamblado del Pipeline de MLlib y lectura de importancias
  modelos.py         submuestreo, catálogo de experimentos, CrossValidator, ALS
  evaluacion.py      AUC-PR, lift por decil, calibración, métricas de ranking
  seguimiento.py     MLflow: qué se registra y la tabla comparativa
  utilidades.py      cronómetro, tamaños en disco, exportación a JSON
scripts/
  correr_fase2.py             el pipeline completo desde CLI
  generar_datos_sinteticos.py CSV de juguete con los MISMOS defectos que los reales
  generar_notebook.py         arma el notebook DESDE los módulos (una sola fuente de verdad)
  generar_informe.js          arma el .docx DESDE resultados_fase2.json (ninguna cifra a mano)
  figuras_desde_json.py       rehace calibración y curva de aprendizaje desde el JSON, sin Spark
tests/
  test_humo.py       12 pruebas: contrato, split, fuga, recalibración, regla de calidad, pipeline
notebooks/
  fase1_eda_benchmark_v2.ipynb
  fase2_pipeline_ml_mlflow.ipynb
```

## Las cinco decisiones que hay que conocer antes de leer el código

**1. El split es temporal y hay un día de burn-in.**
`06-may` no se entrena: existe para que los features históricos del `07-may` no
sean nulos. Train `07..11-may`, validación `12-may`, test `13-may`, y el test se
toca una sola vez. Un split aleatorio mezclaría impresiones del mismo usuario
entre train y test y regalaría señal que en producción no existe.

**2. Ningún feature puede ver el futuro, y eso se verifica.**
Los CTR por entidad se calculan con retardo de un día y ventana expansiva; los de
usuario, con `rowsBetween(unboundedPreceding, -1)` y el mismo desempate total que
la Fase 1 necesitó para que W2 fuera reproducible; el CTR móvil del slot mantiene
`RANGE BETWEEN 6 PRECEDING AND 1 PRECEDING`. `gold.verificar_fuga()` es un control
ejecutable y el pipeline se detiene si no pasa.

**3. El desbalance se ataca submuestreando negativos, y la probabilidad se recalibra.**
Se conserva 1 de cada 10 negativos y **todos** los positivos. La probabilidad
resultante está inflada y se corrige con `CorrectorPrior`
(`p = p_s·r / (p_s·r + 1 − p_s)`). La evaluación va **siempre** sobre el conjunto
completo: AUC-PR depende de la prevalencia y medirlo sobre la muestra daría un
número que no existe en producción.

**4. `brand` no es un predictor.**
El cast fabricó 246.330 nulos que no existen en el origen (Fase 1 §2). Usarla
sería modelar un artefacto de la línea de carga. Sobrevive como `brand_conocida`.

**5. Lo informativo se decide con una prueba, no con un umbral.**
`decidir_imputacion()` usa una prueba de dos proporciones. Un umbral fijo sobre la
razón de CTR está mal calibrado a esta escala: el grupo sin perfil difiere en un
4% relativo —por debajo de cualquier umbral razonable— y sin embargo esa
diferencia tiene **z = 11** sobre 1,5M de observaciones.

## Resultados de la corrida (dataset completo)

| | |
|---|---|
| Mejor modelo | `lr_completo` · **AUC-PR 0,09353** contra un piso de 0,05032 (**1,86×**) |
| *Lift* del decil superior | **2,22×** el CTR global (11,15% contra 5,03%) |
| Aporte de la historia diferida | **+48%** de AUC-PR sobre el mismo modelo sin ella (0,06321) |
| Submuestreo vs `weightCol` | empate en AUC-PR (−0,0001) a **1/4,6 del tiempo** |
| Curva de aprendizaje | **plana**: 10× más filas → +0,23% de AUC-PR |
| ALS vs popularidad | **pierde**: MAP@10 0,1755 contra 0,2194, y cubre 55,1% contra 100% |
| Pipeline completo | 3.097,8 s · 51,6 min · US$ 0,2323 de cómputo equivalente |

## Uso de IA generativa

Declarado en la sección 13 del notebook y en el informe, con el detalle de dónde se
usó, con qué se validó y qué correcciones hizo el equipo sobre lo que produjo el
asistente.

## Continuidad con la Fase 1

La capa Bronze es la misma y se **reutiliza** si ya existe en disco y cumple el
contrato (`bronze.construir` lo verifica en vez de suponerlo). Las decisiones que
la Fase 1 dejó pendientes —qué hacer con `pvalue_level` y `new_user_class_level`—
se resuelven en `silver.decidir_imputacion()` con la evidencia medida, no por
costumbre.
