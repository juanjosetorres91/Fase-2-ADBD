# Cómo cerrar la entrega de la Fase 2

**Entrega: lunes 21 de septiembre, 23:59 vía Canvas** (informe) **+ repositorio** (código).

El notebook ya corrió de punta a punta sobre los 26.557.961 registros reales: **37/37 celdas, cero
errores, 51,6 min** en una estación de 22 núcleos. Este documento dice qué queda.

| Pieza | Estado |
|---|---|
| `fase2_pipeline_ml_mlflow.ipynb` | **Corrido.** Actualizado con una corrección de fondo (abajo) que pide una última pasada. |
| Paquete `adbd/` + `scripts/` + `tests/` | **Listo.** 12 pruebas en verde, incluida una nueva sobre la regla corregida. |
| `informe_fase2_ADBD.docx` | **Completo, 8 páginas, con las cifras reales.** Cero huecos. |
| Guion de la defensa oral | Pendiente — conviene escribirlo después de la Fase 3, porque el panel evalúa todos los módulos. |

---

## Lo único que falta: una última corrida

Se encontró **un error de fondo** al revisar los resultados, y está arreglado en el código de este
repositorio. No cambia ningún número del modelo, pero sí cambia un argumento del informe, así que
conviene que el notebook entregado lo refleje.

**Qué estaba mal.** `decidir_imputacion()` decidía si la ausencia de un dato era informativa con un
umbral fijo: *"informativa si el CTR difiere más de un 5%"*. Sobre 26,6M de filas ese umbral está mal
calibrado. El grupo sin perfil tiene **5,335%** de CTR contra **5,132%** del grupo con perfil —una
razón de 1,040, por debajo del umbral— y sin embargo la diferencia tiene **z = 11,0** sobre 1,5M de
observaciones: es real más allá de cualquier duda. Con la regla vieja, las ocho columnas de perfil
salieron marcadas *"ausencia no informativa"*, **contradiciendo el hallazgo de la Fase 1** de que ese
segmento clickea por encima del promedio.

**Cómo se arregló.** La regla ahora usa una prueba de dos proporciones y reporta `z`. El tamaño del
efecto se sigue mostrando, aparte, para que el lector juzgue si además importa. La decisión final de
las ocho columnas **no cambia** (`categoria_desconocido`), pero el motivo pasa de *"no informativa"*
a *"la ausencia ES un dato"*, que es lo correcto y lo que el informe argumenta.

Lo detectó un resultado que no encajaba con lo ya sabido, no un error de ejecución: el notebook
corrió sin fallas y la tabla salió mal argumentada.

```bash
# 1. Restart & Run All una vez más (~52 min). Todo lo demás es determinista:
#    misma semilla, mismos splits, mismos modelos. Solo cambian los tiempos.

# 2. Regenerar el informe desde el JSON nuevo (30 s)
node scripts/generar_informe.js resultados_fase2.json informe_fase2_ADBD.docx
```

Si por tiempo deciden **no** volver a correrlo, el informe que se adjunta ya trae la tabla corregida
y es defendible; lo único que quedaría desalineado es la celda 20 del notebook entregado. En la
defensa eso se cuenta como lo que es —una corrección posterior a la corrida— y suma en vez de restar.

## Las tres cosas que revisar antes de mandar

1. **El control de fuga dio `OK` en las tres pruebas** (máx |r| = 0,0828 en `ctr_usuario_previo`).
   Confirmado en la corrida.
2. **La línea base da AUC-PR = 0,05032, idéntico al CTR observado del día de test.** Ese cuadre es el
   control de que la evaluación está bien montada. Confirmado.
3. **Las importancias del ganador.** Ninguna variable concentra el peso (la mayor es `log_price` con
   20,8%), así que no hay señal de fuga. Confirmado — pero ver la nota de abajo.

## Lo que la corrida cambió respecto del borrador

Tres conclusiones **se invirtieron o se concretaron** con los datos reales, y las tres están ya
escritas así en el informe:

- **ALS pierde contra la popularidad.** MAP@10 0,1755 contra 0,2194, y encima cubre solo el 55,1% de
  los usuarios contra el 100%. Es un resultado negativo legítimo y se reporta como tal, con el
  diagnóstico: a nivel de `cate_id` la granularidad es demasiado gruesa para que la personalización
  pague.
- **La curva de aprendizaje está plana.** 10× más filas dan +0,23% de AUC-PR. La recomendación sobre
  `behavior_log` es **no traerlo**, con el límite explícito de que eso vale para este espacio de
  variables: `behavior_log` aportaría *features* nuevas, no más filas de las mismas.
- **El submuestreo no cuesta precisión.** `weightCol` sobre las 16,7M de filas da AUC-PR 0,093432
  contra 0,093534 del submuestreo sobre 2,46M, y tarda **4,6× más**. El atajo compra cómputo gratis.

Y una que conviene tener a mano para el panel: **los coeficientes de `ctr_hist_*` quedaron fuera del
top 15**, aunque el bloque de historia diferida sea lo que da el +48% de AUC-PR. El peso se lo
llevaron `ctr_movil_6h_slot` y los agregados de volumen. La lectura honesta es que `m = 200` puede
estar suavizando de más las tasas por entidad; es la primera iteración pendiente y es barata.

## Lo que queda armado para la Fase 3

- **Atributos protegidos ya en Gold** (`final_gender_code`, `age_level`, `new_user_class_level`), con
  el segmento sin perfil conservado y marcado — y ahora con la evidencia de que esa ausencia es
  informativa, que es en sí un dato para el capítulo de equidad.
- **El modelo entrenado y registrado en MLflow** (`lr_completo`, run `c1a542e6…`), que es sobre el que
  se miden disparidad de impacto y paridad demográfica.
- **La línea base de equidad de la Fase 1**: a igual edad y nivel de ciudad, el género 1 ve avisos
  1,76× más caros con CTR *menor*. Esa disparidad **ya está en el histórico**, así que el modelo la
  hereda en vez de introducirla.
- **`ts_local` por segundo sobre 8 días continuos**, listo para el *replay* a Kafka con ventanas y
  *watermarking*.
- **El costo con la misma tarifa** (US$ 0,27/hora) y la advertencia de que los tiempos de la Fase 1
  (2 núcleos) y los de la Fase 2 (22 núcleos) **no son comparables entre sí** — algo que hay que
  repetir en el capítulo FinOps de la Fase 3 si se vuelve a cambiar de máquina.
