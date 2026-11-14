"""Predicción de riesgo de deserción con el modelo entrenado en scripts/entrenar_modelo_riesgo.py."""
from __future__ import annotations

import json

import joblib
import pandas as pd

from .config import RAIZ

BANDAS = [(0.6, "alto"), (0.3, "medio"), (0.0, "bajo")]


class ModeloRiesgo:
    def __init__(self):
        ruta_modelo = RAIZ / "models" / "modelo_riesgo.joblib"
        try:
            paquete = joblib.load(ruta_modelo)
        except Exception:
            # Versión de scikit-learn distinta a la del entrenamiento: se reentrena (tarda ~1 s)
            import subprocess
            import sys
            subprocess.run([sys.executable, str(RAIZ / "scripts" / "entrenar_modelo_riesgo.py")], check=True)
            paquete = joblib.load(ruta_modelo)
        self.modelo, self.variables, self.nombre = paquete["modelo"], paquete["variables"], paquete["nombre"]
        ruta = RAIZ / "resultados" / "metricas_modelo.json"
        self.metricas = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}

    def predecir(self, valores: dict) -> dict:
        x = pd.DataFrame([[float(valores[c]) for c in self.variables]], columns=list(self.variables))
        p = float(self.modelo.predict_proba(x)[0, 1])
        banda = next(n for lim, n in BANDAS if p >= lim)
        return {"probabilidad": p, "banda": banda, "contribuciones": self._contribuciones(x)}

    def _contribuciones(self, x) -> list[dict]:
        """Aporte de cada variable al logit (coeficiente × valor estandarizado) en la regresión logística."""
        clf = self.modelo.named_steps.get("clf")
        pre = self.modelo.named_steps.get("pre")
        if not hasattr(clf, "coef_") or pre is None:
            return []
        z = pre.transform(x)[0]
        aportes = clf.coef_[0] * z
        filas = [{"variable": self.variables[c], "aporte": float(a)} for c, a in zip(self.variables, aportes)]
        return sorted(filas, key=lambda d: -abs(d["aporte"]))


def pregunta_orientacion(valores: dict, resultado: dict) -> str:
    """Construye la consulta que se envía al RAG según los factores de alerta del estudiante."""
    temas = []
    if valores["Tuition fees up to date"] == 0 or valores["Debtor"] == 1:
        temas.append("dificultades para pagar la matrícula o deudas con la institución")
    inscritas = max(1.0, valores["Curricular units 1st sem (enrolled)"])
    if valores["Curricular units 1st sem (approved)"] / inscritas < 0.5:
        temas.append("bajo rendimiento académico y pérdida de asignaturas")
    if valores["Scholarship holder"] == 0 and resultado["banda"] != "bajo":
        temas.append("acceso a becas o apoyos económicos")
    if valores["Age at enrollment"] >= 25:
        temas.append("estudiantes que trabajan o ingresan con mayor edad")
    if not temas:
        temas.append("acompañamiento y bienestar para mantenerse en la carrera")
    return ("Soy un estudiante con riesgo de deserción por " + ", ".join(temas) +
            ". ¿Qué opciones me da el reglamento y qué estrategias de apoyo recomiendan el MEN y bienestar?")


def tabla_tasas() -> dict:
    ruta = RAIZ / "data" / "tasas_por_factor.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
