"""Genera docs/informe.pdf a partir de resultados/resultados.json."""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

RAIZ = Path(__file__).resolve().parent.parent
R = json.loads((RAIZ / "resultados" / "resultados.json").read_text(encoding="utf-8"))
SALIDA = RAIZ / "docs"
SALIDA.mkdir(exist_ok=True)

NEGRO = colors.black
GRIS = colors.HexColor("#eeeeee")

base = getSampleStyleSheet()
titulo = ParagraphStyle("T", parent=base["Title"], textColor=NEGRO, fontSize=22, leading=26)
sub = ParagraphStyle("S", parent=base["Normal"], alignment=1, textColor=colors.black)
h1 = ParagraphStyle("H1", parent=base["Heading1"], textColor=NEGRO, fontSize=15, spaceBefore=14, spaceAfter=6, keepWithNext=1)
h2 = ParagraphStyle("H2", parent=base["Heading2"], textColor=NEGRO, fontSize=12, spaceBefore=8, spaceAfter=4, keepWithNext=1)
cuerpo = ParagraphStyle("C", parent=base["Normal"], fontSize=10.5, leading=15, alignment=TA_JUSTIFY, spaceAfter=6)
viñeta = ParagraphStyle("V", parent=cuerpo, leftIndent=14, bulletIndent=4, spaceAfter=3)


def p(t):
    return Paragraph(t, cuerpo)


def v(t):
    return Paragraph(t, viñeta, bulletText="•")


def tabla(filas, anchos):
    t = Table(filas, colWidths=anchos, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d0d0d0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GRIS]),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return t


def pct(x):
    return f"{x * 100:.1f} %".replace(".", ",")


def dec(x):
    return f"{x:.3f}".replace(".", ",")


m, b = R["regresion_logistica"], R["modelo_base"]
mc = R["matriz_confusion"]
cv = R["validacion_cruzada_f1"]
d = R["datos"]

s = []
s += [Spacer(1, 3 * cm), Paragraph("Clasificación de tumores de mama con regresión logística", titulo)]
s += [Paragraph("Mini-proyecto de Machine Learning: evaluación y análisis ético", sub), Spacer(1, 6)]
s += [Paragraph("Autor: Bmoya29 · Rama: dev_Bmoya29 · Repositorio: proyectoFinal_diagnostico_cancer", sub)]
s += [Spacer(1, 1.2 * cm)]

s += [Paragraph("Resumen", h1)]
s += [
    p(
        f"Se entrenó una regresión logística para predecir si un tumor de mama es benigno o maligno a partir de "
        f"{d['variables']} medidas numéricas. Sobre {R['prueba']} muestras de prueba el modelo alcanzó una exactitud de "
        f"{pct(m['accuracy'])}, frente a {pct(b['accuracy'])} del modelo de referencia que siempre responde benigno. "
        f"El recall de la clase maligna fue {pct(m['recall'])}: de {mc['VP'] + mc['FN']} tumores malignos, el modelo "
        f"dejó pasar {mc['FN']}. Los resultados son prometedores para un ejercicio académico, pero no justifican "
        f"su uso clínico."
    )
]

s += [Paragraph("Modelo seleccionado", h1)]
s += [
    p(
        "Se eligió la regresión logística, un modelo de clasificación que calcula la probabilidad de que un caso "
        "pertenezca a la clase maligna y decide según un umbral (0,5 por defecto). Se escogió por cuatro razones:"
    ),
    v("Es simple y rápida, adecuada para un dataset de 569 muestras."),
    v("Entrega probabilidades, no solo una etiqueta, lo que permite ajustar el umbral según el costo de cada error."),
    v("Sus coeficientes son interpretables, algo importante cuando la decisión afecta a una persona."),
    v("Es menos propensa al sobreajuste que modelos más flexibles cuando hay pocos datos."),
    p(
        "Como punto de comparación se usó un DummyClassifier que siempre predice la clase más frecuente (benigno). "
        "Si la regresión logística no lo supera con claridad, no estaría aprendiendo nada útil."
    ),
]

s += [Paragraph("Dataset utilizado", h1)]
s += [
    p(
        f"Se usó el dataset Breast Cancer Wisconsin (Diagnostic), disponible en scikit-learn y originado en el "
        f"repositorio UCI. Contiene {d['filas']} muestras y {d['variables']} variables numéricas calculadas a partir de "
        f"imágenes digitalizadas de biopsias con aguja fina. Para cada núcleo celular se miden diez características "
        f"(radio, textura, perímetro, área, suavidad, compacidad, concavidad, puntos cóncavos, simetría y dimensión "
        f"fractal) y de cada una se reporta la media, el error estándar y el peor valor."
    ),
    p(
        f"El objetivo del modelo es predecir el diagnóstico. En este proyecto la clase positiva es maligno "
        f"({d['malignos']} casos, {str(d['porcentaje_malignos']).replace('.', ',')} %) y la negativa es benigno "
        f"({d['benignos']} casos). Los datos se dividieron de forma estratificada: {R['entrenamiento']} muestras para "
        f"entrenar y {R['prueba']} para probar. Las variables se estandarizaron dentro de un Pipeline para que el "
        f"escalado se aprendiera solo con los datos de entrenamiento y no hubiera fuga de información."
    ),
]

s += [Paragraph("Métricas de evaluación", h1)]
s += [
    tabla(
        [
            ["Métrica", "Modelo de referencia", "Regresión logística"],
            ["Accuracy", dec(b["accuracy"]), dec(m["accuracy"])],
            ["Precision", dec(b["precision"]), dec(m["precision"])],
            ["Recall", dec(b["recall"]), dec(m["recall"])],
            ["F1-score", dec(b["f1"]), dec(m["f1"])],
        ],
        [5 * cm, 5 * cm, 5 * cm],
    ),
    Spacer(1, 8),
    p("Interpretación de cada métrica, tomando maligno como clase positiva:"),
    v(
        f"<b>Accuracy ({dec(m['accuracy'])}):</b> el modelo acierta {pct(m['accuracy'])} de los casos. Por sí sola "
        f"engaña un poco, porque el modelo de referencia ya obtiene {pct(b['accuracy'])} solo por el desbalance de clases."
    ),
    v(
        f"<b>Precision ({dec(m['precision'])}):</b> de cada 100 casos que el modelo marca como malignos, unos "
        f"{round(m['precision'] * 100)} lo son realmente. Mide qué tan confiable es una alarma."
    ),
    v(
        f"<b>Recall ({dec(m['recall'])}):</b> de cada 100 tumores malignos, el modelo detecta unos "
        f"{round(m['recall'] * 100)}. Es la métrica más crítica aquí, porque no detectar un cáncer es el error más grave."
    ),
    v(
        f"<b>F1-score ({dec(m['f1'])}):</b> resume precision y recall en un solo número; baja si alguno de los dos es débil."
    ),
    p(
        f"La validación cruzada de 5 particiones sobre el conjunto de entrenamiento dio un F1 medio de "
        f"{dec(cv['media'])} con desviación de {dec(cv['desviacion'])}, coherente con el resultado en prueba."
    ),
]

s += [Paragraph("Matriz de confusión", h1)]
s += [
    Image(str(RAIZ / "resultados" / "matriz_confusion.png"), width=9 * cm, height=8.1 * cm),
    Spacer(1, 6),
    tabla(
        [
            ["", "Predicho benigno", "Predicho maligno"],
            ["Real benigno", f"{mc['VN']} (VN)", f"{mc['FP']} (FP)"],
            ["Real maligno", f"{mc['FN']} (FN)", f"{mc['VP']} (VP)"],
        ],
        [4 * cm, 5 * cm, 5 * cm],
    ),
    Spacer(1, 8),
    v(f"<b>Verdaderos positivos ({mc['VP']}):</b> tumores malignos que el modelo identificó correctamente."),
    v(
        f"<b>Falsos positivos ({mc['FP']}):</b> tumores benignos que el modelo marcó como malignos. "
        f"Generan angustia y pruebas adicionales innecesarias, pero se corrigen con estudios posteriores."
    ),
    v(f"<b>Verdaderos negativos ({mc['VN']}):</b> tumores benignos identificados correctamente."),
    v(
        f"<b>Falsos negativos ({mc['FN']}):</b> tumores malignos que el modelo clasificó como benignos. "
        f"Es el error más costoso, porque puede retrasar un tratamiento."
    ),
]

s += [Paragraph("Variables más influyentes", h1)]
tm = R["top_variables_maligno"]
s += [
    p(
        "Los coeficientes del modelo, calculados sobre variables estandarizadas, indican qué medidas empujan la "
        "predicción hacia maligno. Las cinco con mayor peso positivo fueron: "
        + ", ".join(f"{k} ({str(v_).replace('.', ',')})" for k, v_ in tm.items())
        + ". Esto da una explicación global del modelo, pero no explica por qué se clasificó un caso concreto, y "
        "las variables están correlacionadas entre sí, por lo que cada coeficiente debe leerse con cautela."
    ),
    Image(str(RAIZ / "resultados" / "variables_influyentes.png"), width=12 * cm, height=7.7 * cm),
]

s += [Paragraph("Análisis ético", h1)]
s += [
    Paragraph("Sesgos del dataset y del modelo", h2),
    p(
        "El dataset proviene de una sola institución y época, y no incluye variables demográficas como edad, origen "
        "étnico o antecedentes familiares. No se puede verificar si la muestra representa a la población general ni "
        "medir el desempeño por subgrupos. Si ciertos grupos están poco representados, el modelo podría funcionar "
        "peor con ellos sin que las métricas globales lo muestren. Además, las etiquetas dependen del criterio de "
        "los médicos que diagnosticaron, por lo que cualquier error humano queda incorporado en los datos."
    ),
    Paragraph("Equidad", h2),
    p(
        "Un modelo con buenas métricas promedio puede ser injusto si falla más en un grupo concreto. Antes de un uso "
        "real habría que evaluar recall y precision por grupo de edad, origen y tipo de equipo de captura, y "
        "comprobar que el costo de los errores no recae de forma desigual sobre unas personas."
    ),
    Paragraph("Privacidad", h2),
    p(
        "Este dataset es público y no contiene identificadores personales, por lo que su uso aquí no expone a nadie. "
        "En un proyecto real con historias clínicas se necesitaría consentimiento informado, anonimización, "
        "almacenamiento seguro y cumplimiento de la normativa de protección de datos de salud aplicable."
    ),
    Paragraph("Impacto social", h2),
    p(
        "Una herramienta así podría apoyar el tamizaje y reducir tiempos de diagnóstico, sobre todo donde hay pocos "
        "especialistas. Pero también podría generar exceso de confianza en la máquina, falsa tranquilidad ante un "
        "falso negativo o desigualdad de acceso si solo se implementa en ciertas instituciones."
    ),
    Paragraph("Justicia, transparencia y uso responsable", h2),
    p(
        "En decisiones de salud, la persona afectada tiene derecho a saber que un algoritmo participó y a "
        "recibir una explicación comprensible. La regresión logística ayuda porque es transparente, pero eso no "
        "basta: el modelo debe usarse solo como apoyo, con un profesional que tome la decisión final y asuma la "
        "responsabilidad. También conviene documentar sus límites, monitorear su desempeño en el tiempo y "
        "reconocer abiertamente que este proyecto es educativo y no un dispositivo médico."
    ),
]

s += [Paragraph("Conclusiones técnicas", h1)]
s += [
    v(
        f"El modelo funcionó bien: superó con amplio margen al modelo de referencia (F1 de {dec(m['f1'])} frente a "
        f"{dec(b['f1'])}) y mantuvo un resultado estable en validación cruzada."
    ),
    v(
        f"La precision ({dec(m['precision'])}) fue mayor que el recall ({dec(m['recall'])}). El modelo es cauteloso al "
        f"declarar malignidad, pero deja pasar {mc['FN']} casos malignos, lo cual es indeseable en este contexto."
    ),
    v(
        f"Con solo {mc['VP'] + mc['FN']} casos malignos en la prueba, un acierto más o menos cambia el recall en más "
        f"de dos puntos porcentuales. Las cifras son una estimación, no una garantía."
    ),
    v("El dataset es limpio y de laboratorio; el desempeño real con datos de otro hospital probablemente sería menor."),
]

s += [Paragraph("Recomendaciones para mejorar", h1)]
s += [
    v("Bajar el umbral de decisión por debajo de 0,5 para aumentar el recall, aceptando más falsos positivos, y medir el costo de cada tipo de error."),
    v("Comparar con otros modelos (árbol de decisión, bosque aleatorio, KNN) usando la misma validación cruzada."),
    v("Evaluar con validación cruzada repetida e intervalos de confianza para reflejar la incertidumbre."),
    v("Conseguir datos de otras instituciones y poblaciones, con variables demográficas, para auditar sesgos por grupo."),
    v("Revisar la correlación entre variables y probar selección de características o regularización más fuerte."),
    v("Añadir explicaciones por caso individual y revisión humana obligatoria antes de cualquier uso real."),
]


def pie(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.black)
    canvas.drawCentredString(letter[0] / 2, 1.2 * cm, f"Bmoya29 · proyectoFinal_diagnostico_cancer · página {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(
    str(SALIDA / "informe.pdf"),
    pagesize=letter,
    leftMargin=2.2 * cm,
    rightMargin=2.2 * cm,
    topMargin=2 * cm,
    bottomMargin=2 * cm,
    title="Clasificación de tumores de mama con regresión logística",
    author="Bmoya29",
)
doc.build(s, onFirstPage=pie, onLaterPages=pie)
print("OK", SALIDA / "informe.pdf")
