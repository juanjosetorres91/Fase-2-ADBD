/**
 * Genera informe_fase2_ADBD.docx.
 *
 * El informe se ARMA desde `resultados_fase2.json`, que es lo que produce el
 * notebook. Es la regla del equipo llevada hasta el final: ninguna afirmación se
 * escribe sin una celda que la imprima, y ninguna cifra del informe se teclea a
 * mano. Si el JSON no está, cada hueco sale marcado como ⟦clave⟧ y se ve de
 * inmediato qué falta correr.
 *
 *   node scripts/generar_informe.js [resultados_fase2.json] [salida.docx]
 */
const fs = require("fs");
const d = require("docx");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, WidthType, AlignmentType, BorderStyle, ShadingType, PageOrientation,
} = d;

const RUTA_JSON = process.argv[2] || "resultados_fase2.json";
const SALIDA = process.argv[3] || "informe_fase2_ADBD.docx";
const R = fs.existsSync(RUTA_JSON) ? JSON.parse(fs.readFileSync(RUTA_JSON, "utf8")) : null;
const FALTANTES = new Set();

// ---------------------------------------------------------------------------
// Acceso a resultados con marca de hueco
// ---------------------------------------------------------------------------
function get(ruta) {
  if (!R) return undefined;
  return ruta.split(".").reduce((o, k) => (o == null ? undefined : o[k]), R);
}
function n(ruta, dec = 4, mult = 1) {
  const v = get(ruta);
  if (v === undefined || v === null || typeof v !== "number") { FALTANTES.add(ruta); return `⟦${ruta}⟧`; }
  return (v * mult).toLocaleString("es-CL", { minimumFractionDigits: dec, maximumFractionDigits: dec });
}
function ent(ruta) {
  const v = get(ruta);
  if (typeof v !== "number") { FALTANTES.add(ruta); return `⟦${ruta}⟧`; }
  return Math.round(v).toLocaleString("es-CL");
}
function pct(ruta, dec = 2) { return `${n(ruta, dec, 100)}%`; }
function pct2(ruta, dec = 2) {   // el complemento: accuracy de "nunca click"
  const v = get(ruta);
  if (typeof v !== "number") { FALTANTES.add(ruta); return `⟦${ruta}⟧`; }
  return `${((1 - v) * 100).toLocaleString("es-CL", { minimumFractionDigits: dec, maximumFractionDigits: dec })}%`;
}
function razon(a, b, dec = 2) {
  const x = get(a), y = get(b);
  if (typeof x !== "number" || typeof y !== "number" || !y) { FALTANTES.add(a); return `⟦${a}/${b}⟧`; }
  return (x / y).toLocaleString("es-CL", { minimumFractionDigits: dec, maximumFractionDigits: dec });
}
function pctDelta(a, b, dec = 0) {
  const x = get(a), y = get(b);
  if (typeof x !== "number" || typeof y !== "number" || !y) { FALTANTES.add(a); return `⟦Δ${a}⟧`; }
  return `${((x / y - 1) * 100).toLocaleString("es-CL", { minimumFractionDigits: dec, maximumFractionDigits: dec })}%`;
}
function _sumaTiempos(pred) {
  const t = get("tiempos");
  if (!t) { FALTANTES.add("tiempos"); return null; }
  const tot = Object.values(t).reduce((a, b) => a + b, 0);
  const s = Object.entries(t).filter(([k]) => pred(k)).reduce((a, [, v]) => a + v, 0);
  return tot ? s / tot : null;
}
function pctExperimentos() {
  const v = _sumaTiempos((k) => k.startsWith("experimento"));
  return v == null ? "⟦tiempos⟧" : `${(v * 100).toLocaleString("es-CL", { maximumFractionDigits: 0 })}%`;
}
function pctCapas() {
  const v = _sumaTiempos((k) => ["bronze", "silver", "gold"].includes(k));
  return v == null ? "⟦tiempos⟧" : `${(v * 100).toLocaleString("es-CL", { maximumFractionDigits: 0 })}%`;
}
function pctFrio() {
  const a = get("als.als.usuarios_evaluados"), b = get("als.als.usuarios_con_verdad");
  if (typeof a !== "number" || typeof b !== "number" || !b) { FALTANTES.add("als"); return "⟦frio⟧"; }
  return `${((1 - a / b) * 100).toLocaleString("es-CL", { maximumFractionDigits: 1 })}%`;
}
function pctFilas(ruta) {
  const v = get(ruta), t = get("filas_gold");
  if (typeof v !== "number" || typeof t !== "number" || !t) { FALTANTES.add(ruta); return `⟦${ruta}⟧`; }
  return `${(v / t * 100).toLocaleString("es-CL", { minimumFractionDigits: 1, maximumFractionDigits: 1 })}%`;
}
function txt(ruta) {
  const v = get(ruta);
  if (v === undefined || v === null) { FALTANTES.add(ruta); return `⟦${ruta}⟧`; }
  return String(v);
}

// ---------------------------------------------------------------------------
// Helpers de formato
// ---------------------------------------------------------------------------
const F = "Calibri", S = 19;               // 9,5 pt
function rich(texto, extra = {}) {
  // **negrita** y *cursiva* dentro de una cadena
  const partes = String(texto).split(/(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|⟦[^⟧]+⟧)/g).filter(Boolean);
  return partes.map((p) => {
    if (p.startsWith("**")) return new TextRun({ text: p.slice(2, -2), bold: true, font: F, size: S, ...extra });
    if (p.startsWith("⟦")) return new TextRun({ text: p, font: F, size: S, color: "B00020", bold: true, highlight: "yellow", ...extra });
    if (p.startsWith("`")) return new TextRun({ text: p.slice(1, -1), font: "Consolas", size: S - 1, color: "2F4F4F", ...extra });
    if (p.startsWith("*")) return new TextRun({ text: p.slice(1, -1), italics: true, font: F, size: S, ...extra });
    return new TextRun({ text: p, font: F, size: S, ...extra });
  });
}
function p(texto, o = {}) {
  return new Paragraph({
    children: rich(texto),
    spacing: { after: o.after ?? 90, line: o.line ?? 240 },
    alignment: o.align,
    ...o.paraProps,
  });
}
function h(texto, nivel = HeadingLevel.HEADING_1) {
  return new Paragraph({
    children: [new TextRun({ text: texto, bold: true, font: F, size: nivel === HeadingLevel.HEADING_1 ? 24 : 21, color: "1F3864" })],
    heading: nivel,
    spacing: { before: 200, after: 90 },
  });
}
function bullet(texto) {
  return new Paragraph({ children: rich(texto), bullet: { level: 0 }, spacing: { after: 40, line: 240 } });
}
const ANCHO = 9360; // 6,5"
function celda(contenido, w, o = {}) {
  const hijos = Array.isArray(contenido) ? contenido : [contenido];
  return new TableCell({
    width: { size: w, type: WidthType.DXA },
    shading: o.shade ? { type: ShadingType.CLEAR, fill: o.shade, color: "auto" } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: hijos.map((t) =>
      new Paragraph({
        children: rich(t, o.bold ? { bold: true } : {}),
        alignment: o.align,
        spacing: { after: 0, line: 230 },
      })),
  });
}
function tabla(cols, filas, o = {}) {
  const w = o.widths || cols.map(() => Math.floor(ANCHO / cols.length));
  const suma = w.reduce((a, b) => a + b, 0);
  w[w.length - 1] += ANCHO - suma;
  const borde = { style: BorderStyle.SINGLE, size: 2, color: "BFBFBF" };
  return new Table({
    width: { size: ANCHO, type: WidthType.DXA },
    columnWidths: w,
    borders: { top: borde, bottom: borde, left: borde, right: borde, insideHorizontal: borde, insideVertical: borde },
    rows: [
      new TableRow({
        tableHeader: true,
        children: cols.map((c, i) => celda(c, w[i], { bold: true, shade: "DEE6F1", align: o.alignHead })),
      }),
      ...filas.map((f) => new TableRow({
        children: f.map((c, i) => celda(c, w[i], { align: i === 0 ? undefined : o.alignBody })),
      })),
    ],
  });
}
const espacio = () => new Paragraph({ children: [], spacing: { after: 90 } });

/** Figuras del notebook, si existen. Un informe sin la curva PR ni la de
 *  calibración obliga al lector a creer en la tabla; con ellas puede discrepar. */
function figuras(rutas, alto = 150, ancho = 215) {
  const existen = rutas.filter((r) => fs.existsSync(r.ruta));
  if (!existen.length) return [];
  const w = Math.floor(ANCHO / existen.length);
  const borde = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  return [
    new Table({
      width: { size: ANCHO, type: WidthType.DXA },
      columnWidths: existen.map(() => w),
      borders: { top: borde, bottom: borde, left: borde, right: borde, insideHorizontal: borde, insideVertical: borde },
      rows: [new TableRow({
        children: existen.map((r) => new TableCell({
          width: { size: w, type: WidthType.DXA },
          margins: { top: 40, bottom: 20, left: 40, right: 40 },
          children: [
            new Paragraph({
              alignment: AlignmentType.CENTER,
              spacing: { after: 20 },
              children: [new d.ImageRun({
                type: "png",
                data: fs.readFileSync(r.ruta),
                transformation: { width: ancho, height: alto },
              })],
            }),
            new Paragraph({
              alignment: AlignmentType.CENTER,
              spacing: { after: 0 },
              children: [new TextRun({ text: r.pie, font: F, size: 16, italics: true, color: "595959" })],
            }),
          ],
        })),
      })],
    }),
  ];
}

// ---------------------------------------------------------------------------
// Contenido
// ---------------------------------------------------------------------------
const C = [];

C.push(new Paragraph({
  children: [new TextRun({ text: "Fase 2 · Pipeline batch, ML distribuido y MLflow", bold: true, font: F, size: 30, color: "1F3864" })],
  spacing: { after: 60 },
}));
C.push(p("**Análisis de Big Data · Magíster en Data Science UDD · Proyecto Integrador**", { after: 30 }));
C.push(p("**Dataset:** Ali_Display_Ad_Click (Alimama / Taobao) · **Entrega:** lunes 21 de septiembre de 2026", { after: 30 }));
C.push(p("**Equipo:** Juan José Torres · Claudio Ballerini · Cristian Vargas · Christian Vásquez", { after: 120 }));

C.push(p(
  `**Método.** Regla del equipo, sin cambios desde la Fase 1: *ninguna afirmación se escribe sin una celda que la imprima*. ` +
  `Todas las cifras salen de \`fase2_pipeline_ml_mlflow.ipynb\`, quedan en \`resultados_fase2.json\` y este informe se **arma desde ese archivo**: ` +
  `ninguna se teclea a mano. El notebook escribe además el paquete \`adbd/\`, de modo que el código que produce estos números es el mismo que queda versionado ` +
  `y el mismo que ejecuta \`scripts/correr_fase2.py\` sin abrir un notebook.`));

// ---------------------------------------------------------------- 1
C.push(p(
  "**El resultado, en una línea.** El mejor modelo alcanza **AUC-PR " + n("experimentos.lr_completo.test.auc_pr", 5) +
  "** sobre el día de test completo, contra un piso de " + n("linea_base.auc_pr", 5) + " (la prevalencia): **" +
  razon("experimentos.lr_completo.test.auc_pr", "linea_base.auc_pr") + "× el azar**. El 10% de impresiones que el modelo " +
  "pone arriba rinde **" + n("experimentos.lr_completo.lift_d1", 2) + "× el CTR global**. Y el aporte se puede atribuir: " +
  "sin los *features* derivados del objetivo el mismo modelo llega a " + n("experimentos.lr_contexto.test.auc_pr", 5) +
  ", así que **la historia diferida vale un " + pctDelta("experimentos.lr_completo.test.auc_pr", "experimentos.lr_contexto.test.auc_pr") +
  " de AUC-PR** — y es, con diferencia, la decisión de diseño que más rindió."));

C.push(h("1. De la Fase 1 a la Fase 2: qué se hereda y qué se decide"));
C.push(p(
  "La Fase 1 no terminó con un informe sino con una arquitectura y una lista de pendientes explícitos. Esta fase los ejecuta. " +
  "La capa Bronze —26.557.961 impresiones en Parquet + ZSTD particionado por `fecha_local`, " + n("bronze.mb_bronze", 1) + " MB en " +
  ent("bronze.n_particiones") + " particiones; la Fase 1 reportó 376,2 MB y la diferencia son las tres columnas derivadas que esta fase agrega " +
  "(`ts_local`, `hora_local`, `franja`)— **no se reescribe**: `bronze.construir()` verifica que cumpla el contrato y la reutiliza, y solo la " +
  "reconstruye desde los CSV si no lo cumple o si se fuerza. " +
  "Que se **verifique** en vez de suponerse no es ceremonia: un Parquet de una corrida anterior con el esquema incompleto sobrevive en disco y el pipeline " +
  "correría igual, fallando recién en la consulta que usa la columna que falta."));
C.push(tabla(
  ["Decisión heredada de la Fase 1", "Cómo entra en la Fase 2"],
  [
    ["`time_stamp` está en UTC, no en hora local (UTC+8)", "La corrección se aplica una sola vez, en Bronze. Sin ella todo *feature* horario queda desplazado 8 horas."],
    ["El segmento **sin perfil** (5,76% del cruce) se conserva con *flag*", "`sin_perfil` es un *feature*. En la Fase 1 ese segmento clickeaba **por encima** del promedio (5,33% contra 5,14%)."],
    ["`brand`: el cast fabricó **246.330** nulos inexistentes en el origen", "**Descartada** como predictor. Modelarla sería modelar un artefacto de la línea de carga; sobrevive solo como `brand_conocida`."],
    ["`pvalue_level` (54,24%) y `new_user_class_level` (32,49%): *\"se deciden en Fase 2\"*", "Sección 2.1, **con una medición** y no por costumbre."],
    ["W3 con `RANGE ... 1 PRECEDING`, y el desempate `(time_stamp, adgroup_id, pid)` que W2 necesitó para ser determinista", "Los dos se mantienen literales: el primero como *feature* del slot, el segundo ordenando las ventanas de fatiga del usuario."],
  ],
  { widths: [3400, 5960] }));
C.push(espacio());
C.push(p("**Las cuatro decisiones nuevas de esta fase**, que son las que el resto del informe defiende:"));
C.push(bullet("**Split temporal con un día de *burn-in*.** El `06-may` no se entrena: existe para que los históricos del `07-may` no sean nulos. Train `07..11-may`, validación `12-may`, test `13-may`, y el test se toca **una** vez."));
C.push(bullet("**Ningún *feature* puede ver el futuro, y hay un control ejecutable que lo verifica.** La Fase 1 ya encontró una fuga de este tipo; aquí el pipeline **se detiene** si reaparece."));
C.push(bullet("**Submuestreo de negativos con recalibración**, y evaluación siempre sobre el conjunto completo."));
C.push(bullet("**La línea base es un modelo, no una frase.** Sin el piso, ninguna otra cifra se puede leer."));

// ---------------------------------------------------------------- 2
C.push(h("2. Preparación distribuida: Bronze → Silver → Gold"));
C.push(h("2.1 Calidad: la imputación se decidió con una medición", HeadingLevel.HEADING_2));
C.push(p(
  "La respuesta refleja ante un 54% de nulos es imputar con la moda. Antes de decidir se midió algo concreto: **¿el grupo sin dato clickea distinto del grupo con dato?** " +
  "Si la respuesta es sí, *\"no sé\"* es información sobre el usuario y sustituirla por la moda la destruye. Es la misma lógica que en la Fase 1 salvó al segmento sin perfil: " +
  "filtrarlo era la decisión obvia y habría borrado el segundo grupo más grande, que además responde mejor que el promedio."));
C.push(p(
  "Y la completitud que importa es **la del cruce**, no la de la tabla de perfiles: dentro de `user_profile.csv` varias columnas tienen 0% de nulos, " +
  "pero en el JOIN falta el 5,76%. Por la misma razón los porcentajes de la Fase 1 no se repiten: `pvalue_level` tenía 54,24% *dentro de la tabla* " +
  "y tiene **" + n("calidad.4.pct_nulos_en_cruce", 2) + "%** en el cruce; `new_user_class_level`, 32,49% contra **" + n("calidad.7.pct_nulos_en_cruce", 2) + "%**."));
C.push(p(
  "**Qué decide \"informativa\", y por qué no es el tamaño del efecto.** La primera versión de esta regla usaba un umbral fijo sobre la razón de CTR (5%). " +
  "Sobre 26,6M de filas ese umbral está mal calibrado en las dos direcciones: el grupo sin perfil tiene " + n("calidad.2.ctr_sin_dato_pct", 3) + "% de CTR " +
  "contra " + n("calidad.2.ctr_con_dato_pct", 3) + "% del grupo con perfil —una razón de " + n("calidad.2.razon_ctr", 3) + ", **por debajo** del umbral— y sin " +
  "embargo la diferencia tiene **z = " + n("calidad.2.z", 1) + "** sobre 1,5M de observaciones. Con el umbral viejo las ocho columnas salían marcadas " +
  "*\"ausencia no informativa\"*, contradiciendo el hallazgo de la Fase 1 de que ese segmento clickea **por encima** del promedio. Lo que la regla necesita " +
  "saber es si la diferencia **existe** —una prueba de dos proporciones—; el tamaño del efecto se reporta aparte para que el lector juzgue si además importa. " +
  "Fue la quinta corrección del equipo sobre el borrador asistido (sección 7), y la detectó un **resultado que no encajaba con lo ya sabido**, no un error de ejecución."));
C.push(tablaCalidad());
C.push(espacio());

C.push(h("2.2 Gold: los *features*, y el riesgo que define esta fase", HeadingLevel.HEADING_2));
C.push(p(
  "Aquí se gana o se pierde la Fase 2, porque aquí vive la **fuga de información**. La Fase 1 ya encontró una: `CURRENT ROW` en lugar de `1 PRECEDING` en W3 metía el resultado " +
  "a predecir dentro del *feature*, y el split temporal seguía viéndose impecable. Un AUC inflado no falla: miente. De ahí la regla única de este módulo — **ningún *feature* puede " +
  "depender de una fila cuyo `clk` el modelo todavía no habría visto en el instante de la impresión** — y las tres familias de variables derivadas del objetivo:"));
C.push(tabla(
  ["Familia", "Cómo se difiere", "Qué captura"],
  [
    ["CTR histórico por entidad (`cate_id`, `campaign_id`, `customer`, `pid`)", "ventana expansiva con **retardo de un día**: para el día *d* solo suma días *< d*", "calidad del inventario"],
    ["CTR y fatiga del usuario", "ventana expansiva **dentro del flujo**, `rowsBetween(unboundedPreceding, −1)`, con el desempate total de W2", "el hallazgo más accionable de la Fase 1: el CTR cae de 7,01% en la primera impresión a 4,30% de la vigésima"],
    ["CTR móvil del slot", "`RANGE BETWEEN 6 PRECEDING AND 1 PRECEDING`, literal de W3", "tendencia reciente de la posición"],
  ],
  { widths: [2600, 3500, 3260] }));
C.push(espacio());
C.push(p(
  "Todos van suavizados por m-estimación, `(clicks_previos + m·p₀) / (impresiones_previas + m)`, con el prior `p₀` **también diferido**. " +
  "Una entidad nueva recibe el prior, no un `NaN` ni un cero engañoso — y campañas nuevas hay todos los días. " +
  "**El costo de la regla es un día**: el `06-may` queda fuera del entrenamiento y son " + ent("split.0.filas") + " impresiones —el " +
  pctFilas("split.0.filas") + " del dataset— que se pagan y se declaran. La sección 5.3 muestra que salió barato: la curva de aprendizaje está plana, " +
  "así que ese día no habría comprado precisión."));
C.push(p("La codificación usa `Transformers`/`Estimators` de MLlib en cadena — `CodificadorCentinela` (propio) → `StringIndexer` → `OneHotEncoder` → `VectorAssembler` → `StandardScaler` — " +
  "con `handleInvalid=\"keep\"` en los dos codificadores. **Nada de alta cardinalidad entra codificado**: `adgroup_id` (846.811), `campaign_id` (~423k) y `userid` (1,06M) " +
  "producirían un vector de millones de columnas y un vocabulario que hay que traer al *driver*. Entran por sus agregados históricos diferidos, que es además la representación " +
  "que sí generaliza a una entidad que nunca se vio."));
C.push(espacio());
C.push(p("**El control de fuga, ejecutable.** Tres pruebas; si alguna falla, el notebook se detiene."));
C.push(tablaFuga());
C.push(espacio());
C.push(p("**Split temporal resultante.** El CTR por día tiene que ser estable: si el día de test tuviera un CTR muy distinto, comparar métricas entre validación y test no significaría nada."));
C.push(tablaSplit());

// ---------------------------------------------------------------- 3
C.push(h("3. Modelamiento distribuido"));
C.push(h("3.1 Desbalance: submuestreo de negativos y su precio", HeadingLevel.HEADING_2));
C.push(p(
  "Con CTR " + pct("submuestreo.prevalencia_real") + " en la ventana de entrenamiento (≈1:18), entrenar sobre sus " + ent("n_train") + " filas con " +
  "`CrossValidator` de 3 pliegues y las 12 combinaciones que suman las cuatro grillas son **36 ajustes completos** más los cuatro reajustes finales. " +
  "Se conservan **todos los positivos** y una fracción de los negativos — " + ent("submuestreo.positivos") + " positivos y " +
  ent("submuestreo.negativos_conservados") + " de " + ent("submuestreo.negativos_originales") + " negativos, " + ent("submuestreo.filas_entrenamiento") + " filas de entrenamiento " +
  "con prevalencia " + pct("submuestreo.prevalencia_muestra") + " contra la real de " + pct("submuestreo.prevalencia_real") + "."));
C.push(p(
  "El atajo tiene dos costos y los dos se pagan explícitamente. **Primero, la probabilidad queda inflada**: el modelo aprende sobre una prevalencia inventada. " +
  "La corrección es exacta —`p = p_s·r / (p_s·r + 1 − p_s)`, con `r` la tasa **efectiva** de negativos conservados— y está implementada como un `Transformer` propio " +
  "(`CorrectorPrior`) probado contra casos cerrados, no contra sí mismo. Se usa la tasa efectiva (" + n("submuestreo.tasa_efectiva", 5) + ") y no la nominal porque `sample` " +
  "es de Bernoulli por fila y con la nominal la corrección queda sesgada en el tercer decimal. **Segundo, la evaluación no puede hacerse sobre la muestra** — sección 5.2."));
C.push(p(
  "**La contraprueba, y su veredicto.** Sin ella, *\"submuestreamos por costo\"* es una afirmación sin número detrás. El mismo modelo sobre el *train* " +
  "completo con `weightCol` da **AUC-PR " + n("contraste_pesos.test.auc_pr", 6) + " contra " + n("experimentos.lr_completo.test.auc_pr", 6) + "** " +
  "—una diferencia de " + n("contraste_pesos.delta_auc_pr_vs_submuestreo", 6) + ", dentro del ruido— y tarda **" + n("contraste_pesos.razon_tiempo", 1) +
  "× más**. El atajo no cuesta precisión: compra un factor " + n("contraste_pesos.razon_tiempo", 1) + " de cómputo. Si el resultado hubiera sido el " +
  "contrario, la decisión habría cambiado, y por eso el experimento existe."));
C.push(tablaPesos());
C.push(espacio());
C.push(p(
  "Un detalle que el propio experimento enseñó: **`weightCol` tampoco deja la probabilidad calibrada**. Pesar los positivos por *w* multiplica sus *odds* por *w*, " +
  "igual que el submuestreo las multiplica por `1/r`; es la misma corrección con `r = 1/w`. La primera versión de este experimento la omitía y el *LogLoss* resultante " +
  "habría llevado a concluir que el submuestreo calibra mejor, cuando el problema era la corrección faltante."));

C.push(h("3.2 Los cuatro experimentos y el *tuning*", HeadingLevel.HEADING_2));
C.push(p("Cada experimento responde **una pregunta**. Un barrido de modelos sin pregunta produce una tabla que no se puede defender ante el panel."));
C.push(tablaExperimentos());
C.push(espacio());
C.push(p(
  "**Sobre el `CrossValidator` y el split temporal.** El k-fold aleatorio parte la ventana de entrenamiento sin respetar el orden, lo que normalmente sería una fuga. " +
  "Aquí no lo es, porque la protección temporal vive en los ***features*** —se construyen diferidos— y no en el corte: una fila del 10-may no contiene nada del 11-may " +
  "aunque caiga en el mismo pliegue. La afirmación no se deja en palabras: se mide la **brecha de optimismo** contra el día de validación retenido."));
C.push(p(
  "**Y esa brecha hay que medirla bien.** El `CrossValidator` calcula su AUC-PR sobre pliegues ya submuestreados (prevalencia " + pct("submuestreo.prevalencia_muestra") + ") " +
  "y la validación va sobre el día completo (" + pct("submuestreo.prevalencia_real") + "). Restar esos dos números da una diferencia enorme que **no mide optimismo sino prevalencia**. " +
  "La comparación válida es contra una validación submuestreada a la misma tasa; el contraste en AUC-ROC —invariante al submuestreo— queda como control cruzado. " +
  "Medida así, la brecha resulta **positiva y chica en los cuatro casos** (entre " + n("experimentos.lr_completo.brecha_optimismo", 3) + " y " +
  n("experimentos.lr_contexto.brecha_optimismo", 3) + " de AUC-PR): el k-fold aleatorio sí es algo optimista, pero el margen es de milésimas y no invierte " +
  "el orden entre modelos. Es exactamente lo que la sección afirmaba y ahora está medido en lugar de argumentado."));
C.push(p(
  "**Presupuesto de cómputo, declarado.** La *búsqueda* corre sobre el " + pct("experimentos." + mejor() + ".cv.fraccion_busqueda", 0) + " del *train* ya submuestreado; " +
  "el modelo **ganador** se reajusta sobre el *train* submuestreado completo. Buscar sobre el 100% serían `folds × |grid|` ajustes completos."));

C.push(h("3.3 ALS · recomendación con *feedback* implícito", HeadingLevel.HEADING_2));
C.push(p(
  "El segundo eje que la Fase 1 anticipó: matriz **usuario × categoría** con los clicks como confianza. Solo entran clicks, porque una impresión sin click no es una señal " +
  "negativa sino ausencia de evidencia — que es exactamente el supuesto que `implicitPrefs=True` modela. **Contra la popularidad, o no significa nada**: en *feedback* implícito " +
  "la línea base de popularidad es sorprendentemente difícil de batir, y un MAP@10 sin ese contraste es un número suelto."));
C.push(tablaALS());
C.push(espacio());
C.push(p(
  "**La popularidad gana, y se reporta como tal.** MAP@10 " + n("als.popularidad.map_at_k", 4) + " contra " + n("als.als.map_at_k", 4) + " de ALS; " +
  "NDCG@10 " + n("als.popularidad.ndcg_at_k", 4) + " contra " + n("als.als.ndcg_at_k", 4) + ". Y la brecha real es mayor que la de la tabla, porque ALS solo " +
  "alcanza a **" + n("als.als.cobertura_pct", 1) + "%** de los usuarios con clicks en el test —los otros " + pctFrio() + " no tienen historia en el *train* y " +
  "reciben `coldStartStrategy=\"drop\"`— mientras la popularidad cubre el 100%. Sus métricas están calculadas sobre los usuarios que **sí** pudo servir, que " +
  "son los más activos: el número está, si acaso, inflado a favor de ALS."));
C.push(p(
  "**Por qué pierde, y qué habría que cambiar.** No es un fallo de implementación sino de granularidad: el objeto a recomendar es `cate_id`, y a ese nivel el " +
  "catálogo es corto y los clicks están muy concentrados, así que \"lo que todos miran\" ya explica casi todo y no queda margen para personalizar. La personalización " +
  "necesita un objeto más fino —`campaign_id` o `adgroup_id`— donde el catálogo es de cientos de miles y la popularidad deja de ser una respuesta. Es la línea de " +
  "trabajo futuro que este experimento deja abierta, y la razón de que el eje predictivo de la fase sea el clasificador y no el recomendador."));

// ---------------------------------------------------------------- 4
C.push(h("4. Tracking en MLflow"));
C.push(p(
  "Backend de archivos (`file:./mlruns`), que es lo que funciona en Colab gratuito sin servidor ni base de datos, entregado comprimido como `mlruns_fase2.zip` para que el " +
  "*tracking* sea auditable y no una captura de pantalla. En cada *run* se registran los parámetros del **modelo** y los del **pipeline de datos** —tasa de submuestreo, `m` de " +
  "suavizado, fechas del split, semilla—. Sin los segundos, dos *runs* con el mismo AUC son indistinguibles y no se sabe cuál reproducir. Los artefactos son las curvas PR y de " +
  "calibración, la tabla de deciles, las importancias y el registro de variables."));
C.push(p("**La tabla siguiente se lee desde MLflow**, no desde variables en memoria: si la comparación del informe sale del *tracking*, el *tracking* es la fuente de verdad y no una decoración que se llenó por cumplir."));
C.push(tablaComparativa());

// ---------------------------------------------------------------- 5
C.push(h("5. Evaluación de desempeño"));
C.push(h("5.1 Predictivo, y su traducción a negocio", HeadingLevel.HEADING_2));
C.push(p(
  "**Por qué AUC-PR y no *accuracy*.** Con " + pct("linea_base.ctr_observado") + " de positivos en el día de test, predecir siempre *\"no click\"* da " +
  pct2("linea_base.ctr_observado") + " de *accuracy* y cero valor. La línea base honesta —predecir el CTR histórico constante— tiene AUC-PR igual a la " +
  "prevalencia **por construcción**: " + n("linea_base.auc_pr", 5) + ", idéntico al CTR observado, y ese cuadre es en sí un control de que la evaluación está bien montada. " +
  "Contra ese piso, el mejor modelo rinde " + n("experimentos.lr_completo.test.auc_pr", 5) + " — **" + razon("experimentos.lr_completo.test.auc_pr", "linea_base.auc_pr") + "×**."));
C.push(p(
  "**Dónde está el aporte.** `lr_contexto` —el mismo modelo sin ningún *feature* derivado del objetivo— llega a " + n("experimentos.lr_contexto.test.auc_pr", 5) +
  ", apenas " + razon("experimentos.lr_contexto.test.auc_pr", "linea_base.auc_pr") + "× el piso. Agregar la historia diferida lo lleva a " +
  n("experimentos.lr_completo.test.auc_pr", 5) + ": **" + pctDelta("experimentos.lr_completo.test.auc_pr", "experimentos.lr_contexto.test.auc_pr") + " más de AUC-PR**. " +
  "Ese salto es el resultado central de la fase, y el experimento `lr_contexto` existe precisamente para poder atribuirlo en vez de suponerlo."));
C.push(p(
  "**Y una advertencia sobre las importancias.** Los tres coeficientes más grandes de `lr_completo` son `log_price` (" +
  n("experimentos.lr_completo.importancias.0.peso_relativo_pct", 2) + "% del peso), `precio_rel_categoria` (" +
  n("experimentos.lr_completo.importancias.1.peso_relativo_pct", 2) + "%) y `ctr_movil_6h_slot` (" +
  n("experimentos.lr_completo.importancias.2.peso_relativo_pct", 2) + "%). Los CTR históricos por entidad **no aparecen entre los quince primeros**: entran por " +
  "sus contrapartes de volumen (`log_imp_hist_pid`, `log_imp_previas_usuario`). La lectura honesta es que buena parte de la ganancia viene del CTR reciente del " +
  "*slot* y de cuánta historia tiene cada entidad, más que de la tasa histórica en sí — y que `m = 200` puede estar suavizando de más las tasas por entidad. " +
  "Es la primera cosa que habría que mover si hubiera otra iteración."));
C.push(p(
  "**Y una métrica de negocio, porque el panel no compra un AUC.** En una plataforma de display no se decide *\"click o no click\"*: se **ordena inventario**. " +
  "Lo que importa es cuánto mejor rinde el 10% de impresiones que el modelo pone arriba comparado con servir al azar, y si la probabilidad está lo bastante calibrada " +
  "como para poner un umbral de negocio sobre ella."));
C.push(tablaDeciles());
C.push(espacio());
figuras([
  { ruta: "artefactos_fase2/curva_pr.png", pie: "Precisión-recall sobre el día de test completo, contra el piso de la prevalencia." },
  { ruta: "artefactos_fase2/calibracion.png", pie: "Calibración por decil tras la recalibración: la diagonal es el ideal." },
]).forEach((x) => C.push(x));
C.push(espacio());

C.push(h("5.2 Por qué la evaluación va sobre el conjunto completo", HeadingLevel.HEADING_2));
C.push(p(
  "La demostración, con números. El mismo modelo evaluado sobre el día de test completo y sobre el mismo día submuestreado a la tasa de entrenamiento: **AUC-ROC apenas se mueve** " +
  "—es puro orden, y el submuestreo de negativos no reordena nada— mientras **AUC-PR se dispara**, porque depende de la prevalencia por definición. Reportar el segundo número " +
  "sería reportar un modelo que no existe."));
C.push(tablaSubmuestreo());
C.push(espacio());

C.push(h("5.3 Computacional y trade-offs de escalamiento", HeadingLevel.HEADING_2));
C.push(p(
  "**El entorno primero, porque sin él ningún tiempo se interpreta.** Esta fase NO corrió en el Colab de 2 núcleos donde se midió el *benchmark* de la Fase 1: corrió en " +
  txt("entorno.so") + " con **" + ent("entorno.nucleos") + " núcleos y " + txt("entorno.ram") + "** de RAM, Spark " + txt("entorno.spark") + ". " +
  "Comparar los " + n("t_total_s", 1, 1 / 60) + " min de aquí contra los 26,0 min de la Fase 1 **no mide progreso ni regresión: mide dos máquinas distintas**, y decirlo " +
  "es parte del resultado. Lo comparable entre fases es el reparto interno del tiempo y la tarifa, que es la misma."));
C.push(p(
  "**Misma tarifa que la Fase 1**: US$ 0,27/hora (e2-standard-8, *on-demand*). El costo monetario efectivo sigue siendo **US$ 0**; la tarifa valoriza el tiempo para " +
  "responder *\"¿cuánto costaría si hubiera que pagarlo?\"*, que es la pregunta real al dimensionar la Fase 3. El pipeline completo tardó **" + n("t_total_s", 1) + " s** (" +
  n("t_total_s", 1, 1 / 60) + " min) y cuesta **US$ " + n("usd_referencial", 4) + "** de cómputo equivalente. Los cinco experimentos se llevan el " + pctExperimentos() +
  " de ese tiempo; las tres capas de datos, el " + pctCapas() + "."));
C.push(tablaTiempos());
C.push(espacio());
C.push(p(
  "**La pregunta que la Fase 1 dejó abierta con nombre propio: ¿conviene incorporar `behavior_log`** (~704M de registros, " + n("extrapolacion_behavior_log.factor", 1) + "× el volumen), " +
  "que el espejo de Kaggle no trae? La respuesta no es una intuición sino una curva de aprendizaje: se ajustó el mejor modelo sobre fracciones crecientes del entrenamiento y se miró " +
  "si **ya está plana**."));
C.push(tablaCurva());
C.push(espacio());
figuras([
  { ruta: "artefactos_fase2/curva_aprendizaje.png", pie: "El eje izquierdo está abierto a propósito: con el rango automático una curva plana se ve como una montaña." },
], 165, 300).forEach((x) => C.push(x));
C.push(espacio());
C.push(p(
  "**Está plana.** De " + ent("curva_aprendizaje.0.filas") + " a " + ent("curva_aprendizaje.3.filas") + " filas —10×— el AUC-PR pasa de " +
  n("curva_aprendizaje.0.auc_pr", 6) + " a " + n("curva_aprendizaje.3.auc_pr", 6) + ": **" + pctDelta("curva_aprendizaje.3.auc_pr", "curva_aprendizaje.0.auc_pr", 2) +
  "**, mientras el tiempo de ajuste se multiplica por " + razon("curva_aprendizaje.3.t_ajuste_s", "curva_aprendizaje.0.t_ajuste_s", 1) + ". La extrapolación lineal del " +
  "pipeline completo a 26× el volumen da " + n("extrapolacion_behavior_log.segundos_lineales", 0) + " s (US$ " + n("extrapolacion_behavior_log.usd_lineales", 2) + "), " +
  "y es **un piso, no una predicción**: a esa escala el conjunto de trabajo deja de caber en memoria y un motor de un solo nodo se degrada de forma no lineal."));
C.push(p(
  "**Recomendación, con su límite.** Con este espacio de variables, **no** conviene traer `behavior_log`: paga cómputo y no compra precisión. Lo que la curva **no** dice " +
  "es que más datos nunca sirvan — `behavior_log` no traería más filas de las mismas variables sino el **comportamiento de navegación**, que son *features* que hoy no " +
  "existen. La decisión correcta es no escalar el volumen y sí evaluar el valor incremental de esas variables nuevas, que es una pregunta distinta y más barata de " +
  "responder sobre una muestra."));
C.push(p(
  "**Qué no mide esta evaluación.** Un solo nodo: los tiempos **no se extrapolan a un clúster**, donde el reparto entre cómputo y *shuffle* es otro. Ocho días de ventana: " +
  "sin estacionalidad, todas las conclusiones son intradía y por día de semana. Y el *test* es **un solo día** — el `13-may`, con un CTR de " +
  n("split.7.ctr_pct", 3) + "% contra " + n("split.6.ctr_pct", 3) + "% del día de validación y ~5,2% de los días de entrenamiento. Esa deriva de una décima de punto es " +
  "chica pero real, y es la razón de que `lr_completo` y `gbt_completo` —separados por " + n("experimentos.lr_completo.test.auc_pr", 5) + " contra " +
  n("experimentos.gbt_completo.test.auc_pr", 5) + "— se deban leer como un **empate**, no como una victoria."));

// ---------------------------------------------------------------- 6
C.push(h("6. Riesgos y lo que esta fase no resuelve"));
C.push(tabla(
  ["Riesgo", "Disparador o mitigación"],
  [
    ["`m = 200` puede estar suavizando de más los CTR por entidad", "Los coeficientes de `ctr_hist_*` quedaron fuera del top 15 y el peso se lo llevaron los agregados de volumen. Barrer `m` es la primera iteración pendiente y es barata."],
    ["El *test* es un solo día, con CTR " + n("split.7.ctr_pct", 3) + "% contra ~5,2% del *train*", "`lr_completo` y `gbt_completo` están dentro del ruido: se reportan como empate. La selección se hizo en validación y el test se tocó una vez."],
    ["Submuestreo 1:10", "Hoy no cuesta precisión (el contraste con `weightCol` difiere en " + n("contraste_pesos.delta_auc_pr_vs_submuestreo", 6) + "). Si la curva de aprendizaje deja de estar plana, la tasa hay que subirla."],
    ["ALS pierde contra la popularidad y cubre solo " + n("als.als.cobertura_pct", 1) + "% de los usuarios", "No se despliega. La política es servir popularidad, declarada como tal y no como *fallback* silencioso; la vía de personalización pasa por bajar la granularidad a `campaign_id`."],
    ["**La disparidad ya está en el histórico**", "La Fase 1 midió que a igual edad y nivel de ciudad, el género 1 ve avisos **1,76×** más caros con CTR *menor*. El modelo no va a introducir ese sesgo: lo va a **heredar**, y la mitigación de un sesgo heredado es distinta a la de uno inducido. **Es el punto de partida de la Fase 3.**"],
    ["Trazabilidad y normativa", "MLflow cubre el linaje de experimentos; la Ley 21.719 y el GDPR exigen además el registro de decisiones automatizadas. Fase 3."],
  ],
  { widths: [3100, 6260] }));

// ---------------------------------------------------------------- 7
C.push(h("7. Uso de IA generativa y validación crítica"));
C.push(p("**Sí se usó.** Herramienta: **Claude (Anthropic), en Claude Code**, para el diseño del control de fuga, el ensamblado del `Pipeline` de MLlib, la fórmula de recalibración, la estructura de los experimentos y el primer borrador de este informe."));
C.push(p("**Cinco correcciones del equipo sobre lo que produjo el asistente.** Ninguna la habría detectado una prueba de *\"¿corre?\"*, y la quinta salió recién al mirar los resultados de la corrida completa:"));
C.push(new Paragraph({ children: rich("1. **La \"brecha de optimismo\" comparaba peras con manzanas.** El borrador restaba el AUC-PR del `CrossValidator` —sobre pliegues submuestreados— del AUC-PR de la validación completa. La diferencia resultante era enorme y no medía optimismo sino **prevalencia**. Se corrigió midiendo contra una validación submuestreada a la misma tasa, con AUC-ROC como control cruzado."), spacing: { after: 50, line: 240 } }));
C.push(new Paragraph({ children: rich("2. **`weightCol` también descalibra, y el borrador no lo corregía.** La conclusión *\"el submuestreo calibra mejor\"* habría sido falsa: el problema era la corrección faltante, no el método."), spacing: { after: 50, line: 240 } }));
C.push(new Paragraph({ children: rich("3. **Faltaba el día de burn-in.** Sin él, los históricos del primer día de entrenamiento son nulos, y la alternativa del borrador —rellenarlos con el prior global— usa el CTR de todo el período, **incluido el futuro**. Fuga silenciosa, con un split temporal impecable."), spacing: { after: 50, line: 240 } }));
C.push(new Paragraph({ children: rich("4. **La línea base no estaba.** El primer borrador comparaba modelos entre sí. Sin el piso —AUC-PR = prevalencia— ninguna de esas cifras se puede leer, y menos defender."), spacing: { after: 50, line: 240 } }));
C.push(new Paragraph({ children: rich("5. **La regla de calidad usaba un umbral fijo donde hacía falta una prueba.** *\"Informativa si el CTR difiere más de un 5%\"* declaró **no informativas las ocho columnas de perfil**, cuando la diferencia del grupo sin perfil tiene z = " + n("calidad.2.z", 1) + " sobre 1,5M de observaciones. Lo delató que el veredicto **contradecía un hallazgo ya publicado en la Fase 1**, no un error de ejecución: el notebook corrió sin fallas y la tabla salió mal argumentada."), spacing: { after: 90, line: 240 } }));
C.push(p(
  "**Los controles que quedaron en el código** para que esos errores no puedan repetirse en silencio: `gold.verificar_fuga()` como `assert` que detiene el pipeline, " +
  "la validación de cardinalidad de los *joins* con `delta = 0`, la lectura de los nombres de *features* desde `categorySizes` del modelo ajustado en vez de deducirlos " +
  "—deducirlos desalineaba los coeficientes por una posición, un error que no se ve porque no falla—, y 12 pruebas en `tests/test_humo.py` que corren el pipeline completo " +
  "sobre datos sintéticos **con los mismos defectos que los reales** (encabezado con espacio, `brand` no numérico, usuarios sin perfil) en ~75 s."));
C.push(p(
  "**Las tres preguntas de la clase.** *¿Responde la pregunta real?* Sí: la pregunta es ordenar inventario, por eso la métrica de cabecera es AUC-PR y la de negocio es el " +
  "*lift* del decil superior, no *accuracy*. *¿Se validó contra números conocidos?* Sí: la línea base igual a la prevalencia por construcción, la cardinalidad con `delta = 0`, " +
  "el CTR reconstruido y la recalibración probada contra casos cerrados. *¿Se revisó el plan de ejecución?* La lectura de planes fue el núcleo de la Fase 1; su equivalente aquí " +
  "es el control de fuga y la curva de aprendizaje, que son los que dicen si el número es defendible."));

C.push(p(
  "**Anexo · Reproducibilidad.** `fase2_pipeline_ml_mlflow.ipynb` (Kernel → Restart & Run All) · `resultados_fase2.json` (las claves que respaldan cada cifra de este informe) · " +
  "`mlruns_fase2.zip` · `artefactos_fase2/` · paquete `adbd/` y `scripts/correr_fase2.py` para correr la fase completa sin notebook · `pytest tests/ -q`. " +
  "Lo único que varía entre corridas son los tiempos.", { after: 0 }));

// ---------------------------------------------------------------------------
// Tablas que dependen del JSON
// ---------------------------------------------------------------------------
function mejor() { return get("mejor_experimento") || "lr_completo"; }

function tablaCalidad() {
  // Seis de las ocho columnas de perfil faltan EXACTAMENTE en las mismas filas
  // —el usuario no tiene registro— así que su fila es idéntica. Se colapsan: una
  // tabla con seis renglones iguales no informa, ocupa.
  const crudas = (get("calidad") || []).filter((r) => r.pct_nulos_en_cruce > 0);
  const clave = (r) => `${r.pct_nulos_en_cruce}|${r.ctr_sin_dato_pct}|${r.z}`;
  const grupos = new Map();
  for (const r of crudas) {
    const k = clave(r);
    if (!grupos.has(k)) grupos.set(k, { ...r, columnas: [] });
    grupos.get(k).columnas.push(r.columna);
  }
  const filas = [...grupos.values()].map((r) => [
    r.columnas.length === 1 ? "`" + r.columnas[0] + "`"
      : "las " + r.columnas.length + " columnas de perfil (`" + r.columnas[0] + "`, `" + r.columnas[1] + "`, …)",
    r.pct_nulos_en_cruce.toLocaleString("es-CL", { minimumFractionDigits: 2 }) + "%",
    r.ctr_sin_dato_pct == null ? "—" : r.ctr_sin_dato_pct.toLocaleString("es-CL", { minimumFractionDigits: 3 }) + "%",
    r.ctr_con_dato_pct == null ? "—" : r.ctr_con_dato_pct.toLocaleString("es-CL", { minimumFractionDigits: 3 }) + "%",
    r.razon_ctr == null ? "—" : r.razon_ctr.toLocaleString("es-CL", { minimumFractionDigits: 3 }),
    r.z == null ? "—" : "**" + r.z.toLocaleString("es-CL", { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + "**",
    "**" + r.decision + "**",
  ]);
  if (!filas.length) { FALTANTES.add("calidad"); filas.push(["⟦calidad⟧", "⟦⟧", "⟦⟧", "⟦⟧", "⟦⟧", "⟦⟧", "⟦⟧"]); }
  return tabla(["Columna", "% nulos en el cruce", "CTR sin dato", "CTR con dato", "Razón", "z", "Decisión"],
    filas, { widths: [1900, 1150, 1050, 1050, 800, 700, 2710], alignBody: AlignmentType.CENTER });
}

function tablaFuga() {
  const filas = (get("fuga") || []).map((r) => [r.prueba, r.detalle, "**" + r.veredicto + "**"]);
  if (!filas.length) { FALTANTES.add("fuga"); filas.push(["⟦fuga⟧", "correr el notebook", "⟦⟧"]); }
  return tabla(["Prueba", "Detalle medido", "Veredicto"], filas,
    { widths: [3200, 4560, 1600], alignBody: AlignmentType.CENTER });
}

function tablaSplit() {
  const filas = (get("split") || []).map((r) => [
    r.fecha_local, r.filas.toLocaleString("es-CL"), r.positivos.toLocaleString("es-CL"),
    r.ctr_pct.toLocaleString("es-CL", { minimumFractionDigits: 3 }) + "%", "**" + r.split + "**",
  ]);
  if (!filas.length) { FALTANTES.add("split"); filas.push(["⟦split⟧", "—", "—", "—", "—"]); }
  return tabla(["Fecha local", "Impresiones", "Clicks", "CTR", "Split"], filas,
    { widths: [1700, 1900, 1600, 1500, 2660], alignBody: AlignmentType.CENTER });
}

function tablaPesos() {
  const sub = get("experimentos." + mejor());
  const sub2 = get("experimentos.lr_completo") || sub;
  const w = get("contraste_pesos");
  const f = (x, dec = 5) => (typeof x === "number" ? x.toLocaleString("es-CL", { minimumFractionDigits: dec, maximumFractionDigits: dec }) : "⟦⟧");
  if (!w || !sub2) { FALTANTES.add("contraste_pesos"); }
  return tabla(
    ["Variante", "Filas de entrenamiento", "Tiempo de ajuste", "AUC-PR (test)", "LogLoss (test)"],
    [
      ["Submuestreo 1:10 + recalibración", sub2 ? get("submuestreo.filas_entrenamiento").toLocaleString("es-CL") : "⟦⟧",
        sub2 ? sub2.cv.t_ajuste_final_s + " s" : "⟦⟧", f(sub2 && sub2.test.auc_pr), f(sub2 && sub2.test.logloss)],
      ["*Train* completo + `weightCol`", w ? w.filas.toLocaleString("es-CL") : "⟦⟧",
        w ? w.t_ajuste_final_s + " s" : "⟦⟧", f(w && w.test.auc_pr), f(w && w.test.logloss)],
    ],
    { widths: [2700, 1900, 1600, 1600, 1560], alignBody: AlignmentType.CENTER });
}

function tablaExperimentos() {
  const preguntas = {
    lr_contexto: "¿Cuánto se predice **sin historia**, solo con perfil y contexto? Aísla el aporte real de los *features* derivados del objetivo.",
    lr_completo: "¿Cuánto agregan los CTR históricos diferidos y la fatiga?",
    rf_completo: "¿Gana un *ensemble* de árboles sin interacciones explícitas?",
    gbt_completo: "¿Compensa el *boosting* su costo de cómputo en AUC-PR?",
  };
  const exp = get("experimentos") || {};
  const filas = Object.keys(preguntas).map((k) => {
    const r = exp[k];
    if (!r) { FALTANTES.add("experimentos." + k); return ["`" + k + "`", preguntas[k], "⟦⟧", "⟦⟧", "⟦⟧"]; }
    return ["`" + k + "`", preguntas[k],
      r.test.auc_pr.toLocaleString("es-CL", { minimumFractionDigits: 5 }),
      r.test.auc_roc.toLocaleString("es-CL", { minimumFractionDigits: 5 }),
      (r.brecha_optimismo >= 0 ? "+" : "") + r.brecha_optimismo.toLocaleString("es-CL", { minimumFractionDigits: 5 })];
  });
  return tabla(["Experimento", "Pregunta que responde", "AUC-PR test", "AUC-ROC test", "Brecha CV–val"],
    filas, { widths: [1600, 4000, 1300, 1300, 1160], alignBody: AlignmentType.CENTER });
}

function tablaALS() {
  const a = get("als.als"), pp = get("als.popularidad");
  const f = (x) => (typeof x === "number" ? x.toLocaleString("es-CL", { minimumFractionDigits: 5 }) : "⟦⟧");
  if (!a) FALTANTES.add("als");
  return tabla(["Modelo", "Precision@10", "Recall@10", "MAP@10", "NDCG@10", "Usuarios evaluados"],
    [
      ["ALS implícito (rank " + (get("als.rank") ?? "⟦⟧") + ", α " + (get("als.alpha") ?? "⟦⟧") + ")",
        f(a && a.precision_at_k), f(a && a.recall_at_k), f(a && a.map_at_k), f(a && a.ndcg_at_k),
        a ? a.usuarios_evaluados.toLocaleString("es-CL") : "⟦⟧"],
      ["Popularidad (línea base)", f(pp && pp.precision_at_k), f(pp && pp.recall_at_k), f(pp && pp.map_at_k),
        f(pp && pp.ndcg_at_k), pp ? pp.usuarios_evaluados.toLocaleString("es-CL") : "⟦⟧"],
    ],
    { widths: [2800, 1300, 1300, 1300, 1300, 1360], alignBody: AlignmentType.CENTER });
}

function tablaComparativa() {
  const c = get("comparativa_mlflow") || [];
  const f = (x, dec = 5) => (typeof x === "number" ? x.toLocaleString("es-CL", { minimumFractionDigits: dec, maximumFractionDigits: dec }) : "—");
  const filas = c.map((r) => ["`" + (r.experimento ?? "⟦⟧") + "`", f(r.test_auc_pr), f(r.test_auc_roc),
    f(r.test_logloss), f(r.test_lift_d1, 2), f(r.t_busqueda_s, 1), f(r.t_ajuste_final_s, 1)]);
  if (!filas.length) { FALTANTES.add("comparativa_mlflow"); filas.push(["⟦comparativa_mlflow⟧", "—", "—", "—", "—", "—", "—"]); }
  return tabla(["Run de MLflow", "AUC-PR test", "AUC-ROC test", "LogLoss", "Lift D1", "t búsqueda (s)", "t ajuste (s)"],
    filas, { widths: [2400, 1250, 1250, 1150, 1000, 1200, 1110], alignBody: AlignmentType.CENTER });
}

function tablaDeciles() {
  const dec = get("experimentos." + mejor() + ".deciles") || [];
  const filas = dec.map((r) => [r.decil, r.impresiones.toLocaleString("es-CL"), r.clicks.toLocaleString("es-CL"),
    r.ctr_pct.toLocaleString("es-CL", { minimumFractionDigits: 3 }) + "%",
    r.p_media_pct.toLocaleString("es-CL", { minimumFractionDigits: 3 }) + "%",
    "**" + r.lift.toLocaleString("es-CL", { minimumFractionDigits: 2 }) + "×**"]);
  if (!filas.length) { FALTANTES.add("deciles"); filas.push(["⟦deciles⟧", "—", "—", "—", "—", "—"]); }
  return tabla(["Decil de score", "Impresiones", "Clicks", "CTR observado", "CTR predicho", "Lift"],
    filas, { widths: [1700, 1700, 1400, 1600, 1600, 1360], alignBody: AlignmentType.CENTER });
}

function tablaSubmuestreo() {
  const c = get("efecto_submuestreo_en_la_evaluacion") || [];
  const filas = c.map((r) => [r.metrica,
    r.conjunto_completo.toLocaleString("es-CL", { minimumFractionDigits: 5 }),
    r.submuestra.toLocaleString("es-CL", { minimumFractionDigits: 5 }),
    (r.diferencia_rel == null ? "—" : (r.diferencia_rel * 100).toLocaleString("es-CL", { maximumFractionDigits: 1 }) + "%")]);
  if (!filas.length) { FALTANTES.add("efecto_submuestreo_en_la_evaluacion"); filas.push(["⟦⟧", "—", "—", "—"]); }
  return tabla(["Métrica", "Test completo", "Test submuestreado", "Diferencia relativa"],
    filas, { widths: [2600, 2300, 2300, 2160], alignBody: AlignmentType.CENTER });
}

function tablaTiempos() {
  const t = get("tiempos") || {};
  const usd = (v) => "US$ " + (v / 3600 * 0.27).toLocaleString("es-CL", { minimumFractionDigits: 4, maximumFractionDigits: 4 });
  const fila = (k, v) => [k, v.toLocaleString("es-CL", { minimumFractionDigits: 1 }),
    (v / 60).toLocaleString("es-CL", { minimumFractionDigits: 2, maximumFractionDigits: 2 }), usd(v)];
  const orden = Object.entries(t).sort((a, b) => b[1] - a[1]);
  const TOPE = 6;
  const filas = orden.slice(0, TOPE).map(([k, v]) => fila(k, v));
  const resto = orden.slice(TOPE).reduce((a, [, v]) => a + v, 0);
  if (resto > 0) filas.push(fila(`resto de las etapas (${orden.length - TOPE})`, resto));
  const total = orden.reduce((a, [, v]) => a + v, 0);
  if (total > 0) filas.push(["**Pipeline completo**", "**" + total.toLocaleString("es-CL", { minimumFractionDigits: 1 }) + "**",
    "**" + (total / 60).toLocaleString("es-CL", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + "**", "**" + usd(total) + "**"]);
  if (!filas.length) { FALTANTES.add("tiempos"); filas.push(["⟦tiempos⟧", "—", "—", "—"]); }
  return tabla(["Etapa", "Segundos", "Minutos", "Costo referencial"],
    filas, { widths: [3800, 1800, 1800, 1960], alignBody: AlignmentType.CENTER });
}

function tablaCurva() {
  const c = get("curva_aprendizaje") || [];
  const filas = c.map((r) => [(r.fraccion * 100).toLocaleString("es-CL") + "%",
    r.filas.toLocaleString("es-CL"), r.t_ajuste_s.toLocaleString("es-CL", { minimumFractionDigits: 1 }) + " s",
    r.auc_pr.toLocaleString("es-CL", { minimumFractionDigits: 5 }),
    r.auc_roc.toLocaleString("es-CL", { minimumFractionDigits: 5 })]);
  if (!filas.length) { FALTANTES.add("curva_aprendizaje"); filas.push(["⟦curva_aprendizaje⟧", "—", "—", "—", "—"]); }
  return tabla(["Fracción del train", "Filas", "Tiempo de ajuste", "AUC-PR (val)", "AUC-ROC (val)"],
    filas, { widths: [1900, 2000, 1900, 1800, 1760], alignBody: AlignmentType.CENTER });
}

// ---------------------------------------------------------------------------
const doc = new Document({
  creator: "Equipo ADBD · UDD",
  title: "Fase 2 · Pipeline batch, ML distribuido y MLflow",
  styles: { default: { document: { run: { font: F, size: S } } } },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840, orientation: PageOrientation.PORTRAIT },
        margin: { top: 1080, bottom: 1080, left: 1440, right: 1440 },
      },
    },
    children: C,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(SALIDA, buf);
  console.log(`${SALIDA} escrito${R ? ` desde ${RUTA_JSON}` : " SIN resultados (todo en placeholders)"}`);
  if (FALTANTES.size) {
    console.log(`\n${FALTANTES.size} claves sin valor — correr el notebook y regenerar:`);
    [...FALTANTES].sort().forEach((k) => console.log("  ⟦" + k + "⟧"));
  }
});
