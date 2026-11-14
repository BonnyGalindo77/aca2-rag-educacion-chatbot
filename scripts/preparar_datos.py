"""Regenera los insumos de datos que usa la aplicación a partir del dataset UCI.

Produce, con la misma lógica SQL del ACA 1:
* data/estudiantes.db                 capa SQL (SQLite) con la tabla `estudiantes`
* data/fichas_dataset.json            10 fichas analíticas (texto) por factor de deserción
* data/evidencia_categoria.json       evidencia más fuerte por categoría de riesgo (metadato de los chunks)
* data/tasas_por_factor.json          tablas de deserción por factor (para gráficos de la app)

Uso:  python scripts/preparar_datos.py
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "data"

PROGRAMAS = ["Tecnologías de producción de biocombustibles", "Diseño de animación y multimedia",
             "Servicio social (nocturno)", "Agronomía", "Diseño de comunicación", "Enfermería veterinaria",
             "Ingeniería informática", "Equinocultura", "Administración", "Servicio social", "Turismo",
             "Enfermería", "Higiene oral", "Gestión de publicidad y marketing", "Periodismo y comunicación",
             "Educación básica", "Administración (nocturno)"]
CODIGOS_UCI = [33, 171, 8014, 9003, 9070, 9085, 9119, 9130, 9147, 9238, 9254, 9500, 9556, 9670, 9773, 9853, 9991]
SI_NO = {0: "No", 1: "Sí"}
FACTORES_SQL = {
    "Deudor con la institución": ("debtor", SI_NO, "financiero", "tener deudas pendientes con la institución"),
    "Matrícula al día": ("tuition_fees_up_to_date", SI_NO, "financiero", "tener el pago de la matrícula al día"),
    "Becado": ("scholarship_holder", SI_NO, "financiero", "contar con beca"),
    "Aprobación de asignaturas sem. 1": ("banda_aprob_sem1", None, "academico",
                                         "el porcentaje de asignaturas aprobadas en el primer semestre"),
    "Edad al ingresar": ("banda_edad", None, "socioeconomico_personal", "la edad al momento de ingresar"),
    "Jornada": ("daytime_evening_attendance", {0: "Nocturna", 1: "Diurna"}, "socioeconomico_personal",
                "la jornada de estudio"),
    "Sexo": ("gender", {0: "Mujer", 1: "Hombre"}, "socioeconomico_personal", "el sexo del estudiante"),
    "Desplazado": ("displaced", SI_NO, "socioeconomico_personal", "vivir desplazado de su lugar de origen"),
    "Necesidades educativas especiales": ("educational_special_needs", SI_NO, "socioeconomico_personal",
                                          "tener necesidades educativas especiales"),
    "Programa académico": ("programa", None, "academico", "el programa académico"),
}


def cargar() -> pd.DataFrame:
    df = pd.read_csv(DATOS / "students_dropout.csv")
    df.columns = [c.strip() for c in df.columns]
    mapa = dict(zip(CODIGOS_UCI, PROGRAMAS)) if df["Course"].max() > 17 else dict(zip(range(1, 18), PROGRAMAS))
    df["programa"] = df["Course"].map(mapa)
    df["tasa_aprob_sem1"] = np.where(df["Curricular units 1st sem (enrolled)"] > 0,
                                     df["Curricular units 1st sem (approved)"] / df["Curricular units 1st sem (enrolled)"],
                                     np.nan)
    df["banda_aprob_sem1"] = pd.cut(df["tasa_aprob_sem1"], [-0.01, 0.0, 0.5, 0.99, 1.0],
                                    labels=["0 % aprobado", "1–50 %", "51–99 %", "100 % aprobado"]
                                    ).astype(object).fillna("sin asignaturas inscritas")
    df["banda_edad"] = pd.cut(df["Age at enrollment"], [0, 20, 24, 34, 120],
                              labels=["17–20 años", "21–24 años", "25–34 años", "35 años o más"]).astype(object)
    return df


def fmt(x):
    return f"{x:.1f}".replace(".", ",")


def main():
    df = cargar()
    con = sqlite3.connect(DATOS / "estudiantes.db")
    df.rename(columns=lambda c: re.sub(r"[^0-9a-zA-Z_]+", "_", c).strip("_").lower()).to_sql(
        "estudiantes", con, if_exists="replace", index=False)

    def tasas_por(columna, etiquetas=None):
        t = pd.read_sql(f"""
            SELECT {columna} AS valor, COUNT(*) AS n,
                   ROUND(100.0*SUM(target='Dropout')/COUNT(*), 1)  AS pct_desercion,
                   ROUND(100.0*SUM(target='Graduate')/COUNT(*), 1) AS pct_graduacion,
                   ROUND(100.0*SUM(target='Enrolled')/COUNT(*), 1) AS pct_matriculado
            FROM estudiantes GROUP BY {columna} ORDER BY pct_desercion DESC""", con)
        if etiquetas:
            t["valor"] = t["valor"].map(etiquetas).fillna(t["valor"])
        return t

    tablas = {f: tasas_por(c, e) for f, (c, e, _, _) in FACTORES_SQL.items()}
    global_ = float(tasas_por("'Todos'").pct_desercion[0])

    fichas, evid_cat = [], {}
    for factor, (col, _, categoria, descripcion) in FACTORES_SQL.items():
        t = tablas[factor]
        filas = "; ".join(f"{r.valor}: {fmt(r.pct_desercion)} % de deserción, {fmt(r.pct_graduacion)} % de graduación "
                          f"(n = {r.n})" for r in t.itertuples())
        mayor, menor = t.iloc[0], t.iloc[-1]
        texto = (f"Ficha analítica del dataset UCI Predict Students' Dropout and Academic Success ({len(df)} estudiantes de "
                 f"educación superior). Factor analizado: {factor.lower()}, es decir, {descripcion}. "
                 f"La deserción global de la cohorte es {fmt(global_)} %. Resultados por grupo: {filas}. "
                 f"El grupo con mayor deserción es «{mayor.valor}» ({fmt(mayor.pct_desercion)} %) y el de menor deserción es "
                 f"«{menor.valor}» ({fmt(menor.pct_desercion)} %), una diferencia de "
                 f"{fmt(mayor.pct_desercion - menor.pct_desercion)} puntos porcentuales. "
                 f"Este factor se relaciona con riesgo de tipo {categoria.replace('_', ' ')} en la permanencia estudiantil.")
        evidencia = (f"{factor}: «{mayor.valor}» {fmt(mayor.pct_desercion)} % vs «{menor.valor}» "
                     f"{fmt(menor.pct_desercion)} % de deserción")
        fichas.append({"factor": factor, "categoria": categoria, "variable": col, "texto": texto, "evidencia": evidencia})
        brecha = float(t.pct_desercion.max() - t.pct_desercion.min())
        if brecha > evid_cat.get(categoria, (0, ""))[0]:
            evid_cat[categoria] = (brecha, evidencia)

    (DATOS / "fichas_dataset.json").write_text(json.dumps(fichas, ensure_ascii=False, indent=1), encoding="utf-8")
    (DATOS / "evidencia_categoria.json").write_text(
        json.dumps({k: v[1] for k, v in evid_cat.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
    (DATOS / "tasas_por_factor.json").write_text(json.dumps(
        {"desercion_global_pct": global_, "registros": len(df),
         "tablas": {f: t.to_dict(orient="records") for f, t in tablas.items()}}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"Deserción global {global_} % · {len(fichas)} fichas · evidencia: {list(evid_cat)}")


if __name__ == "__main__":
    main()
