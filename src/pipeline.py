"""
Flujo completo de Machine Learning: diagnostico de cancer de mama.

Etapas del workflow basico:
1. Definir la tarea
2. Recolectar datos
3. Representar la entrada de forma que el modelo la procese
4. Elegir un modelo
5. Entrenar el modelo
6. Evaluar su desempeno
7. Usar el modelo entrenado con datos nuevos

Ejecucion:
    python src/pipeline.py
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEMILLA = 42
RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "resultados"
MODELOS = RAIZ / "modelos"
DATOS = RAIZ / "datos"


def cargar_datos():
    """Etapas 1 y 2: definir la tarea y recolectar los datos."""
    bunch = load_breast_cancer(as_frame=True)
    X = bunch.data
    # En scikit-learn: 0 = maligno, 1 = benigno. Se redefine para que
    # la clase positiva (1) sea "maligno", que es la que importa detectar.
    y = (bunch.target == 0).astype(int).rename("maligno")
    return X, y


def construir_modelo():
    """Etapas 3 y 4: representar la entrada (escalado) y elegir el modelo."""
    return Pipeline(
        [
            ("escalado", StandardScaler()),
            ("modelo", LogisticRegression(max_iter=1000, random_state=SEMILLA)),
        ]
    )


def metricas(y_real, y_pred):
    return {
        "accuracy": accuracy_score(y_real, y_pred),
        "precision": precision_score(y_real, y_pred, zero_division=0),
        "recall": recall_score(y_real, y_pred),
        "f1": f1_score(y_real, y_pred),
    }


def main():
    SALIDA.mkdir(exist_ok=True)
    MODELOS.mkdir(exist_ok=True)
    DATOS.mkdir(exist_ok=True)

    # Etapas 1 y 2
    X, y = cargar_datos()
    pd.concat([X, y], axis=1).to_csv(DATOS / "breast_cancer.csv", index=False)
    resumen_datos = {
        "filas": int(X.shape[0]),
        "variables": int(X.shape[1]),
        "malignos": int(y.sum()),
        "benignos": int((1 - y).sum()),
        "porcentaje_malignos": round(float(y.mean()) * 100, 2),
    }

    # Separacion estratificada: el escalado se aprende SOLO con entrenamiento
    # (dentro del Pipeline), para evitar fuga de informacion.
    X_ent, X_pru, y_ent, y_pru = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEMILLA
    )

    # Etapa 5: entrenar modelo base y modelo propuesto
    base = DummyClassifier(strategy="most_frequent")
    base.fit(X_ent, y_ent)

    modelo = construir_modelo()
    modelo.fit(X_ent, y_ent)

    # Etapa 6: evaluar
    pred_base = base.predict(X_pru)
    pred_modelo = modelo.predict(X_pru)
    res_base = metricas(y_pru, pred_base)
    res_modelo = metricas(y_pru, pred_modelo)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    cv_f1 = cross_val_score(construir_modelo(), X_ent, y_ent, cv=cv, scoring="f1")

    cm = confusion_matrix(y_pru, pred_modelo)
    vn, fp, fn, vp = (int(v) for v in cm.ravel())

    # Matriz de confusion (figura)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(
        cm, display_labels=["Benigno", "Maligno"]
    ).plot(ax=ax, cmap="Greens", colorbar=False)
    ax.set_title("Matriz de confusion - Regresion logistica")
    fig.tight_layout()
    fig.savefig(SALIDA / "matriz_confusion.png", dpi=160)
    plt.close(fig)

    # Variables mas influyentes (coeficientes sobre datos escalados)
    coef = pd.Series(
        modelo.named_steps["modelo"].coef_[0], index=X.columns
    ).sort_values()
    top = pd.concat([coef.head(5), coef.tail(5)])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colores = ["#2e7d32" if v < 0 else "#c62828" for v in top.values]
    ax.barh(top.index, top.values, color=colores)
    ax.set_title("Variables mas influyentes (coeficientes)")
    ax.set_xlabel("Coeficiente (rojo: empuja hacia maligno)")
    fig.tight_layout()
    fig.savefig(SALIDA / "variables_influyentes.png", dpi=160)
    plt.close(fig)

    # Etapa 7: usar el modelo con datos nuevos (simulados con 3 casos de prueba)
    joblib.dump(modelo, MODELOS / "modelo_regresion_logistica.joblib")
    nuevos = X_pru.iloc[:3]
    probas = modelo.predict_proba(nuevos)[:, 1]
    ejemplos = pd.DataFrame(
        {
            "probabilidad_maligno": np.round(probas, 4),
            "prediccion": np.where(probas >= 0.5, "maligno", "benigno"),
            "real": np.where(y_pru.iloc[:3] == 1, "maligno", "benigno"),
        }
    )
    ejemplos.to_csv(SALIDA / "ejemplos_prediccion.csv", index=False)

    resultados = {
        "datos": resumen_datos,
        "entrenamiento": int(len(X_ent)),
        "prueba": int(len(X_pru)),
        "modelo_base": res_base,
        "regresion_logistica": res_modelo,
        "validacion_cruzada_f1": {
            "media": float(cv_f1.mean()),
            "desviacion": float(cv_f1.std()),
        },
        "matriz_confusion": {"VN": vn, "FP": fp, "FN": fn, "VP": vp},
        "top_variables_maligno": coef.tail(5)[::-1].round(3).to_dict(),
        "top_variables_benigno": coef.head(5).round(3).to_dict(),
    }
    with open(SALIDA / "resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    print(json.dumps(resultados, indent=2, ensure_ascii=False))
    print(ejemplos.to_string(index=False))


if __name__ == "__main__":
    main()
