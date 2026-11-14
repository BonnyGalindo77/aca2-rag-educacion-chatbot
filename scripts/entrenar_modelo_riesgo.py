"""Modelo de alerta temprana de deserción (complemento analítico del chatbot RAG).

Clasificación binaria Dropout vs. no Dropout (Graduate + Enrolled) con 12 variables disponibles al
cierre del primer semestre. Se comparan tres modelos con validación cruzada estratificada de 5
pliegues y se evalúa el ganador en un conjunto de prueba independiente (20 %).

Salidas:
* models/modelo_riesgo.joblib        pipeline de scikit-learn listo para predecir
* resultados/metricas_modelo.json    métricas de CV y de prueba, matriz de confusión e importancias

Uso:  python scripts/entrenar_modelo_riesgo.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parent.parent
SEMILLA = 42

VARIABLES = {  # columna del dataset → etiqueta en español
    "Age at enrollment": "Edad al ingresar",
    "Gender": "Sexo (1 = hombre)",
    "Scholarship holder": "Becado",
    "Debtor": "Deudor con la institución",
    "Tuition fees up to date": "Matrícula al día",
    "Displaced": "Desplazado",
    "Daytime/evening attendance": "Jornada diurna",
    "Admission grade": "Puntaje de admisión",
    "Curricular units 1st sem (enrolled)": "Asignaturas inscritas sem. 1",
    "Curricular units 1st sem (evaluations)": "Evaluaciones presentadas sem. 1",
    "Curricular units 1st sem (approved)": "Asignaturas aprobadas sem. 1",
    "Curricular units 1st sem (grade)": "Promedio sem. 1 (0–20)",
}


def main():
    df = pd.read_csv(RAIZ / "data" / "students_dropout.csv")
    df.columns = [c.strip() for c in df.columns]
    X = df[list(VARIABLES)].astype(float)
    y = (df["Target"] == "Dropout").astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEMILLA)

    escalar = ColumnTransformer([("num", StandardScaler(), list(VARIABLES))])
    candidatos = {
        "Regresión logística": Pipeline([("pre", escalar), ("clf", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEMILLA))]),
        "Random Forest": Pipeline([("clf", RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=5, class_weight="balanced", n_jobs=-1,
            random_state=SEMILLA))]),
        "Gradient Boosting (HGB)": Pipeline([("clf", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, class_weight="balanced", random_state=SEMILLA))]),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    comparacion = []
    for nombre, modelo in candidatos.items():
        s = cross_validate(modelo, X_tr, y_tr, cv=cv, scoring=["roc_auc", "f1", "recall", "precision", "accuracy"])
        comparacion.append({"modelo": nombre, **{m: round(float(np.mean(s[f"test_{m}"])), 4)
                                                for m in ("roc_auc", "f1", "recall", "precision", "accuracy")},
                            "roc_auc_std": round(float(np.std(s["test_roc_auc"])), 4)})
        print(f"{nombre:<26} AUC {comparacion[-1]['roc_auc']:.3f} · F1 {comparacion[-1]['f1']:.3f}")

    ganador = max(comparacion, key=lambda r: (round(r["roc_auc"], 3), r["f1"]))["modelo"]
    modelo = candidatos[ganador].fit(X_tr, y_tr)
    proba = modelo.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    prueba = {"roc_auc": roc_auc_score(y_te, proba), "pr_auc": average_precision_score(y_te, proba),
              "f1": f1_score(y_te, pred), "recall": recall_score(y_te, pred),
              "precision": precision_score(y_te, pred), "accuracy": accuracy_score(y_te, pred)}
    prueba = {k: round(float(v), 4) for k, v in prueba.items()}
    imp = permutation_importance(modelo, X_te, y_te, scoring="roc_auc", n_repeats=10, random_state=SEMILLA)
    importancias = sorted(({"variable": VARIABLES[c], "importancia": round(float(m), 4)}
                           for c, m in zip(VARIABLES, imp.importances_mean)), key=lambda d: -d["importancia"])

    (RAIZ / "models").mkdir(exist_ok=True)
    joblib.dump({"modelo": modelo, "variables": VARIABLES, "nombre": ganador}, RAIZ / "models" / "modelo_riesgo.joblib",
                compress=3)
    salida = {"objetivo": "Dropout (1) vs. Graduate/Enrolled (0)", "registros": int(len(df)),
              "tasa_positiva": round(float(y.mean()), 4), "variables": list(VARIABLES.values()),
              "comparacion_cv5": comparacion, "modelo_seleccionado": ganador, "prueba_20pct": prueba,
              "matriz_confusion": confusion_matrix(y_te, pred).tolist(), "importancia_permutacion": importancias,
              "umbral_decision": 0.5, "bandas_riesgo": {"bajo": [0, 0.3], "medio": [0.3, 0.6], "alto": [0.6, 1.0]}}
    (RAIZ / "resultados").mkdir(exist_ok=True)
    (RAIZ / "resultados" / "metricas_modelo.json").write_text(json.dumps(salida, ensure_ascii=False, indent=1),
                                                             encoding="utf-8")
    print(f"Ganador: {ganador} · prueba: {prueba}")


if __name__ == "__main__":
    main()
