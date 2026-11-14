"""Ingesta de nuevos documentos desde la interfaz.

Replica la canalización del ACA 1 para que los documentos cargados desde la app queden con la misma
limpieza, el mismo chunking (RecursiveCharacterTextSplitter, 1.500 caracteres, 15 % de solapamiento)
y el mismo esquema de payload que el corpus base.
"""
from __future__ import annotations

import io
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from .config import RAIZ

SEPARADORES = ["\nCAPÍTULO", "\nCapítulo", "\nARTÍCULO", "\nArtículo", "\n\n", "\n", ". ", "; ", ", ", " ", ""]

CATEGORIAS = {
    "financiero": (["deuda", "pago", "matricula", "financ", "beca", "credito educativo", "paz y salvo", "costo",
                    "apoyo economico", "subsidio", "icetex", "derechos pecuniarios", "descuento"],
                   ["debtor", "tuition_fees_up_to_date", "scholarship_holder"]),
    "academico": (["asignatura", "calificacion", "nota", "evaluacion", "rendimiento", "reprob", "tutoria",
                   "creditos academicos", "promedio", "nivelacion", "monitoria", "validacion", "habilitacion"],
                  ["curricular_units_1st_sem_approved", "curricular_units_1st_sem_grade", "course"]),
    "socioeconomico_personal": (["familia", "edad", "trabaj", "discapacidad", "vulnerab", "desplaz", "genero",
                                 "etnic", "necesidades educativas", "condicion socioeconomica", "madre", "padre"],
                                ["age_at_enrollment", "displaced", "educational_special_needs", "gender",
                                 "marital_status"]),
    "bienestar_institucional": (["bienestar", "salud mental", "psicolog", "acompanamiento", "consejeria",
                                 "orientacion", "deporte", "cultur", "convivencia", "desarrollo humano"], []),
    "normativo": (["articulo", "acuerdo", "reglamento", "sancion", "derecho", "deber", "reintegro",
                   "transferencia", "homologa", "retiro", "cancelacion", "aplazamiento"], []),
}


def _evidencia() -> dict:
    ruta = RAIZ / "data" / "evidencia_categoria.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


def sin_tildes(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")


def etiquetar(texto: str, evidencia: dict | None = None):
    evidencia = _evidencia() if evidencia is None else evidencia
    t = sin_tildes(texto)
    cats = [c for c, (claves, _) in CATEGORIAS.items() if sum(t.count(k) for k in claves) >= 2]
    variables = sorted({v for c in cats for v in CATEGORIAS[c][1]})
    return cats, variables, [evidencia[c] for c in cats if c in evidencia]


def limpiar_paginas(paginas: list[str]):
    lineas_por_pag = [set(l.strip() for l in p.splitlines() if l.strip()) for p in paginas]
    conteo = Counter(l for s in lineas_por_pag for l in s)
    umbral = max(3, 0.4 * len(paginas))
    repetidas = {l for l, c in conteo.items() if c >= umbral and len(l) < 120}
    limpias = []
    for p in paginas:
        lineas = [l for l in p.splitlines() if l.strip() and l.strip() not in repetidas
                  and not re.fullmatch(r"\s*(p[áa]g(ina)?\.?\s*)?\d{1,3}\s*", l, flags=re.I)]
        t = "\n".join(lineas)
        t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
        t = re.sub(r"[ \t ]+", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        limpias.append(t.strip())
    return limpias


def extraer_paginas(contenido: bytes, nombre: str) -> list[tuple[int, str]]:
    """Devuelve [(número de página, texto limpio)] para PDF, TXT o MD."""
    if nombre.lower().endswith(".pdf"):
        from pypdf import PdfReader

        crudas = [(p.extract_text() or "") for p in PdfReader(io.BytesIO(contenido)).pages]
        limpias = limpiar_paginas(crudas)
        return [(i + 1, t) for i, t in enumerate(limpias) if len(t) >= 80]
    texto = contenido.decode("utf-8", errors="ignore")
    return [(0, re.sub(r"\n{3,}", "\n\n", texto).strip())]


def slug(texto: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", sin_tildes(texto)).strip("_")[:48] or "documento"


def fragmentar(meta: dict, paginas: list[tuple[int, str]], chunk_size: int = 1500, overlap: int = 225,
               modelo: str = "gemini-embedding-2") -> list[dict]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, separators=SEPARADORES,
                                              add_start_index=True, strip_whitespace=True)
    texto, inicios = "", []
    for num, t in paginas:
        inicios.append((len(texto), num))
        texto += t + "\n\n"
    evidencia = _evidencia()
    chunks = []
    for i, d in enumerate(splitter.create_documents([texto])):
        ini = d.metadata["start_index"]
        pagina = max((n for p, n in inicios if p <= ini), default=0)
        cats, variables, evid = etiquetar(d.page_content, evidencia)
        chunks.append({
            "fuente_id": meta["fuente_id"], "tipo_documento": meta.get("tipo_documento", "documento_cargado"),
            "titulo": meta["titulo"], "entidad": meta.get("entidad", ""), "anio": meta.get("anio"),
            "url": meta.get("url", ""), "pagina": pagina, "chunk_idx": i, "texto": d.page_content,
            "factores_riesgo": cats, "variables_dataset": variables, "evidencia_dataset": evid,
            "n_caracteres": len(d.page_content), "n_tokens_aprox": round(len(d.page_content) / 4),
            "chunk_size": chunk_size, "pilar": "educacion", "modelo_principal": modelo, "origen": "carga_app",
        })
    return chunks


def cargar_corpus_local(ruta: Path | None = None) -> list[dict]:
    """Metadatos del corpus base (los mismos del ACA 1)."""
    ruta = ruta or (RAIZ / "data" / "corpus.json")
    return json.loads(ruta.read_text(encoding="utf-8"))
