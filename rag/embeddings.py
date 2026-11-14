"""Vectorización de consultas y documentos.

Usa exactamente el mismo modelo y el mismo formato del ACA 1 para que los vectores de las
consultas sean comparables con los 610 fragmentos ya indexados en Qdrant Cloud:

* ``gemini-embedding-2`` con salida de 768 dimensiones (Matryoshka) y normalización L2.
* Consulta → ``task: search result | query: …`` (equivalente oficial de RETRIEVAL_QUERY).
* Documento → ``title: … | text: …``.
* Si se configura ``gemini-embedding-001`` se usa ``task_type`` de forma literal.
"""
from __future__ import annotations

import hashlib
import re
import time
from collections import OrderedDict

import numpy as np

ERRORES_TRANSITORIOS = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "DEADLINE")


class GeminiEmbedder:
    def __init__(self, api_key: str, modelo: str = "gemini-embedding-2", dim: int = 768,
                 max_reintentos: int = 5):
        from google import genai

        if not api_key:
            raise ValueError("Falta GOOGLE_API_KEY para generar embeddings con Gemini.")
        self.cliente = genai.Client(api_key=api_key)
        self.modelo = modelo
        self.dim = dim
        self.max_reintentos = max_reintentos
        self.es_v2 = modelo.startswith("gemini-embedding-2")
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()   # caché LRU de consultas
        self.uso = {"peticiones": 0, "reintentos": 0}

    # --- formato de entrada idéntico al ACA 1 -------------------------------------------
    def _preparar(self, texto: str, modo: str, titulo: str | None = None) -> str:
        if not self.es_v2:
            return texto
        if modo == "consulta":
            return f"task: search result | query: {texto}"
        return f"title: {titulo or 'none'} | text: {texto}"

    def _llamar(self, textos: list[str], modo: str) -> list[list[float]]:
        from google.genai import types

        if self.es_v2:
            contenidos = [types.Content(parts=[types.Part.from_text(text=t)]) for t in textos]
            config = types.EmbedContentConfig(output_dimensionality=self.dim)
        else:
            contenidos = textos
            config = types.EmbedContentConfig(
                output_dimensionality=self.dim,
                task_type="RETRIEVAL_QUERY" if modo == "consulta" else "RETRIEVAL_DOCUMENT")
        for intento in range(self.max_reintentos + 1):
            try:
                r = self.cliente.models.embed_content(model=self.modelo, contents=contenidos, config=config)
                self.uso["peticiones"] += 1
                return [e.values for e in r.embeddings]
            except Exception as e:  # reintento exponencial ante cuota u ocupación del servicio
                if any(x in str(e) for x in ERRORES_TRANSITORIOS) and intento < self.max_reintentos:
                    self.uso["reintentos"] += 1
                    time.sleep(min(30, 2 ** intento))
                else:
                    raise

    @staticmethod
    def _normalizar(v) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-12)

    # --- API pública ----------------------------------------------------------------------
    def consulta(self, texto: str) -> np.ndarray:
        clave = hashlib.sha1(f"{self.modelo}|{self.dim}|{texto}".encode()).hexdigest()
        if clave in self._cache:
            self._cache.move_to_end(clave)
            return self._cache[clave]
        v = self._normalizar(self._llamar([self._preparar(texto, "consulta")], "consulta")[0])
        self._cache[clave] = v
        if len(self._cache) > 256:
            self._cache.popitem(last=False)
        return v

    def documentos(self, textos: list[str], titulos: list[str] | None = None, lote: int = 24,
                   pausa: float = 0.8, progreso=None) -> np.ndarray:
        titulos = titulos or [None] * len(textos)
        preparados = [self._preparar(t, "documento", ti) for t, ti in zip(textos, titulos)]
        salida = []
        for n, i in enumerate(range(0, len(preparados), lote)):
            if n:
                time.sleep(pausa)   # respeta los límites por minuto de la capa gratuita
            salida.extend(self._normalizar(v) for v in self._llamar(preparados[i:i + lote], "documento"))
            if progreso:
                progreso(min(1.0, (i + lote) / len(preparados)))
        return np.vstack(salida)


class HashEmbedder:
    """Vectorizador léxico determinista, SOLO para pruebas automáticas sin red.

    No es comparable con los vectores de Gemini guardados en Qdrant Cloud; sirve para validar el
    flujo completo (ingesta → búsqueda → interfaz) en una colección local.
    """

    def __init__(self, dim: int = 768):
        self.dim = dim
        self.modelo = f"hash-{dim}"
        self.uso = {"peticiones": 0, "reintentos": 0}

    def _vec(self, texto: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        t = re.sub(r"\s+", " ", texto.lower())
        palabras = re.findall(r"\w+", t)
        for tok in palabras + [t[i:i + 4] for i in range(max(0, len(t) - 3))]:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 8) % 2 else -1.0
        return v / (np.linalg.norm(v) + 1e-12)

    def consulta(self, texto: str) -> np.ndarray:
        return self._vec(texto)

    def documentos(self, textos, titulos=None, progreso=None, **_) -> np.ndarray:
        return np.vstack([self._vec(t) for t in textos])


def crear_embedder(config):
    if config.proveedor_embeddings == "hash":
        return HashEmbedder(config.embed_dim)
    return GeminiEmbedder(config.google_api_key, config.embed_model, config.embed_dim)
