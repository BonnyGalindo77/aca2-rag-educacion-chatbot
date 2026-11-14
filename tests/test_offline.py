"""Pruebas sin red del flujo completo (ingesta → Qdrant en memoria → RAG → citas).

Ejecutar:  python -m pytest -q tests
"""
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
os.environ.update({"QDRANT_URL": ":memory:", "EMBED_PROVIDER": "hash", "LLM_PROVIDER": "offline",
                   "GOOGLE_API_KEY": "", "SCORE_THRESHOLD": "0.30"})

from rag import Config, crear_sistema, analizar_citas  # noqa: E402
from rag.ingest import extraer_paginas, fragmentar, etiquetar  # noqa: E402

TEXTO = """Artículo 58. PÉRDIDA REITERADA DE ASIGNATURAS: El estudiante que deba cursar una asignatura por tercera vez
solo podrá hacerlo con autorización del Comité de Programa.

Artículo 16. RETIRO DEL ESTUDIANTE: la solicitud de retiro deberá formularse dentro de los primeros quince días
calendario del periodo académico; se descuenta el veinticinco por ciento del valor de la matrícula."""


def _sistema():
    s = crear_sistema(Config.desde_entorno())
    s.store.crear_coleccion()
    meta = {"fuente_id": "reglamento_prueba", "titulo": "Reglamento de prueba", "entidad": "CUN", "anio": 2011,
            "tipo_documento": "reglamento"}
    chunks = fragmentar(meta, extraer_paginas(TEXTO.encode(), "r.txt"), chunk_size=200, overlap=20)
    s.store.insertar(chunks, s.embedder.documentos([c["texto"] for c in chunks]))
    return s


def test_responde_con_citas():
    r = _sistema().responder("¿Qué pasa si curso una asignatura por tercera vez?")
    assert r.estado == "ok" and r.fuentes
    assert r.citas["citadas"] and not r.citas["invalidas"]


def test_se_abstiene_fuera_de_dominio():
    r = _sistema().responder("receta del ajiaco santafereño con guascas", umbral=0.95)
    assert r.estado == "sin_evidencia" and "No encontré" in r.respuesta


def test_inventario_y_borrado_protegido():
    s = _sistema()
    assert s.store.fuentes()[0]["fuente_id"] == "reglamento_prueba"
    assert s.store.eliminar_fuente("reglamento_prueba") > 0
    try:
        s.store.eliminar_fuente("reglamento_cun")
        assert False, "el corpus base debe estar protegido"
    except PermissionError:
        pass


def test_etiquetado_y_citas():
    cats, variables, _ = etiquetar("El estudiante con deuda no podrá pagar la matrícula ni recibir beca", {})
    assert "financiero" in cats and "debtor" in variables
    c = analizar_citas("Según el artículo 58 debes pedir autorización [1]. Además aplica [4].", 2)
    assert c["citadas"] == [1] and c["invalidas"] == [4]
