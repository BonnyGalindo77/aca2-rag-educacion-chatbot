"""Configuración central del sistema RAG.

Las credenciales nunca se escriben en el código. Se leen, en este orden, de:
1. ``st.secrets`` (Streamlit Community Cloud o ``.streamlit/secrets.toml`` local),
2. variables de entorno, cargadas también desde un archivo ``.env`` (python-dotenv).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent.parent
load_dotenv(RAIZ / ".env", override=False)

SECRETOS = ("GOOGLE_API_KEY", "QDRANT_API_KEY", "ADMIN_PASSWORD")


def _leer(nombre: str, defecto: str | None = None) -> str | None:
    """Busca una variable en st.secrets y luego en el entorno, sin imprimir nunca su valor."""
    valor = None
    try:
        import streamlit as st

        if nombre in st.secrets:
            valor = str(st.secrets[nombre])
    except Exception:  # fuera de Streamlit o sin secrets.toml
        pass
    if not valor:
        valor = os.getenv(nombre)
    # Limpia espacios o caracteres invisibles que se cuelan al copiar y pegar llaves
    return valor.strip() if valor else defecto


@dataclass
class Config:
    # Credenciales
    google_api_key: str | None = None
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    admin_password: str | None = None

    # Base vectorial (heredada del ACA 1)
    coleccion: str = "aca1_educacion_permanencia"
    vector: str = "gemini"
    embed_model: str = "gemini-embedding-2"
    embed_dim: int = 768

    # Modelo generativo
    llm_model: str = "gemini-3.5-flash"
    llm_respaldo: tuple = ("gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-flash-latest")
    temperatura: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 1024

    # Recuperación
    top_k: int = 5
    umbral: float = 0.645          # calibrado en el ACA 1 con consultas fuera de dominio
    turnos_memoria: int = 4        # turnos previos que se envían al LLM

    # Proveedores alternos (solo para pruebas sin conexión)
    proveedor_embeddings: str = "gemini"   # gemini | hash
    proveedor_llm: str = "gemini"          # gemini | offline

    # Ingesta (mismos parámetros del ACA 1)
    chunk_size: int = 1500
    chunk_overlap: int = 225

    extras: dict = field(default_factory=dict)

    @classmethod
    def desde_entorno(cls) -> "Config":
        c = cls(
            google_api_key=_leer("GOOGLE_API_KEY") or _leer("GEMINI_API_KEY"),
            qdrant_url=_leer("QDRANT_URL"),
            qdrant_api_key=_leer("QDRANT_API_KEY"),
            admin_password=_leer("ADMIN_PASSWORD"),
        )
        c.coleccion = _leer("QDRANT_COLLECTION", c.coleccion)
        c.vector = _leer("QDRANT_VECTOR_NAME", c.vector)
        c.embed_model = _leer("GEMINI_EMBED_MODEL", c.embed_model)
        c.embed_dim = int(_leer("EMBED_DIM", str(c.embed_dim)))
        c.llm_model = _leer("GEMINI_CHAT_MODEL", c.llm_model)
        c.umbral = float(_leer("SCORE_THRESHOLD", str(c.umbral)))
        c.top_k = int(_leer("TOP_K", str(c.top_k)))
        c.proveedor_embeddings = _leer("EMBED_PROVIDER", c.proveedor_embeddings)
        c.proveedor_llm = _leer("LLM_PROVIDER", c.proveedor_llm)
        return c

    def estado_credenciales(self) -> dict:
        """Indica si cada credencial está cargada, sin revelar su valor."""
        return {
            "GOOGLE_API_KEY": bool(self.google_api_key),
            "QDRANT_URL": bool(self.qdrant_url),
            "QDRANT_API_KEY": bool(self.qdrant_api_key),
            "ADMIN_PASSWORD": bool(self.admin_password),
        }

    def publico(self) -> dict:
        """Configuración sin secretos, apta para mostrar en la interfaz o guardar en resultados."""
        d = asdict(self)
        for k in ("google_api_key", "qdrant_api_key", "admin_password"):
            d.pop(k, None)
        d["qdrant_url"] = "configurada" if self.qdrant_url else "no definida"
        d.pop("extras", None)
        return d
