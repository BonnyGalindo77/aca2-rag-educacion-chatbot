"""Acceso a la base vectorial Qdrant Cloud construida en el ACA 1.

La colección ``aca1_educacion_permanencia`` tiene dos vectores nombrados (``gemini`` 768 d y
``e5`` 384 d), distancia coseno e índices de payload sobre ``fuente_id``, ``tipo_documento``,
``factores_riesgo`` y ``pagina``. Este módulo la consulta, la supervisa y permite añadir o retirar
documentos cargados desde la aplicación sin tocar el corpus base.
"""
from __future__ import annotations

import time
import uuid
from collections import Counter, defaultdict

from qdrant_client import QdrantClient, models

FUENTES_BASE = {"reglamento_cun", "guia_permanencia_men", "bienestar_men", "dataset_uci"}
CAMPOS_RESUMEN = ["fuente_id", "titulo", "entidad", "anio", "tipo_documento", "pagina", "url",
                  "origen", "factores_riesgo", "n_caracteres"]


class VectorStore:
    def __init__(self, url: str | None, api_key: str | None, coleccion: str, vector: str = "gemini",
                 dim: int = 768):
        if not url:
            raise ValueError("Falta QDRANT_URL. Configure el .env o los secretos de Streamlit.")
        if url == ":memory:":
            self.cliente = QdrantClient(location=":memory:")
            self.modo = "memoria (pruebas)"
        elif url.startswith("local:"):
            self.cliente = QdrantClient(path=url.removeprefix("local:"))
            self.modo = "local en disco (pruebas)"
        else:
            self.cliente = QdrantClient(url=url, api_key=api_key, timeout=60)
            self.modo = "Qdrant Cloud"
        self.coleccion = coleccion
        self.vector = vector
        self.dim = dim

    # --- estado del índice ----------------------------------------------------------------
    def ping(self) -> float:
        t0 = time.perf_counter()
        self.cliente.get_collections()
        return (time.perf_counter() - t0) * 1000

    def existe(self) -> bool:
        return self.cliente.collection_exists(self.coleccion)

    def info(self) -> dict:
        i = self.cliente.get_collection(self.coleccion)
        vectores = i.config.params.vectors
        if isinstance(vectores, dict):
            vec = {n: {"dim": p.size, "distancia": str(p.distance.value if hasattr(p.distance, "value") else p.distance)}
                   for n, p in vectores.items()}
        else:
            vec = {"(sin nombre)": {"dim": vectores.size, "distancia": str(vectores.distance)}}
        return {
            "estado": str(i.status.value if hasattr(i.status, "value") else i.status),
            "puntos": self.cliente.count(self.coleccion, exact=True).count,
            "vectores_indexados": i.indexed_vectors_count,
            "segmentos": i.segments_count,
            "vectores": vec,
            "indices_payload": sorted((i.payload_schema or {}).keys()),
            "hnsw": {"m": i.config.hnsw_config.m, "ef_construct": i.config.hnsw_config.ef_construct},
        }

    # --- búsqueda --------------------------------------------------------------------------
    @staticmethod
    def _filtro(filtros: dict | None):
        if not filtros:
            return None
        condiciones = []
        for campo, valor in filtros.items():
            if valor in (None, [], ""):
                continue
            cond = models.MatchAny(any=list(valor)) if isinstance(valor, (list, tuple, set)) \
                else models.MatchValue(value=valor)
            condiciones.append(models.FieldCondition(key=campo, match=cond))
        return models.Filter(must=condiciones) if condiciones else None

    def buscar(self, vector, k: int = 5, filtros: dict | None = None):
        return self.cliente.query_points(
            collection_name=self.coleccion, query=list(map(float, vector)), using=self.vector, limit=k,
            query_filter=self._filtro(filtros), with_payload=True,
            search_params=models.SearchParams(hnsw_ef=128),
        ).points

    # --- exploración documental ------------------------------------------------------------
    def _scroll(self, filtro=None, campos=True):
        puntos, offset = [], None
        while True:
            lote, offset = self.cliente.scroll(self.coleccion, scroll_filter=filtro, limit=256, offset=offset,
                                               with_payload=campos, with_vectors=False)
            puntos.extend(lote)
            if offset is None:
                return puntos

    def fuentes(self) -> list[dict]:
        """Resumen por documento: fragmentos, páginas y categorías de riesgo cubiertas."""
        grupos = defaultdict(list)
        for p in self._scroll(campos=CAMPOS_RESUMEN):
            grupos[p.payload.get("fuente_id", "sin_fuente")].append(p.payload)
        filas = []
        for fid, pl in grupos.items():
            base = pl[0]
            factores = Counter(f for x in pl for f in (x.get("factores_riesgo") or []))
            paginas = [x.get("pagina") or 0 for x in pl]
            filas.append({
                "fuente_id": fid,
                "titulo": "Fichas analíticas del dataset UCI (deserción)" if fid == "dataset_uci" else base.get("titulo", fid), "entidad": base.get("entidad", ""),
                "anio": base.get("anio"), "tipo_documento": base.get("tipo_documento", ""),
                "fragmentos": len(pl), "paginas": max(paginas) if any(paginas) else None,
                "caracteres": sum(x.get("n_caracteres") or 0 for x in pl),
                "factores_riesgo": dict(factores.most_common()),
                "origen": base.get("origen", "corpus ACA 1"), "url": base.get("url", ""),
                "protegido": fid in FUENTES_BASE,
            })
        return sorted(filas, key=lambda f: (-f["protegido"], -f["fragmentos"]))

    def fragmentos(self, fuente_id: str) -> list[dict]:
        pts = self._scroll(self._filtro({"fuente_id": fuente_id}))
        filas = [{"id": str(p.id), **p.payload} for p in pts]
        return sorted(filas, key=lambda x: (x.get("pagina") or 0, x.get("chunk_idx") or 0))

    # --- escritura (protegida desde la interfaz) ------------------------------------------
    def asegurar_indices(self):
        existentes = set((self.cliente.get_collection(self.coleccion).payload_schema or {}).keys())
        for campo, tipo in [("fuente_id", models.PayloadSchemaType.KEYWORD),
                            ("tipo_documento", models.PayloadSchemaType.KEYWORD),
                            ("factores_riesgo", models.PayloadSchemaType.KEYWORD),
                            ("pagina", models.PayloadSchemaType.INTEGER),
                            ("origen", models.PayloadSchemaType.KEYWORD)]:
            if campo not in existentes:
                self.cliente.create_payload_index(self.coleccion, field_name=campo, field_schema=tipo)

    def crear_coleccion(self, dim_e5: int = 384):
        """Crea la colección con el mismo esquema del ACA 1 (se usa en modo local de pruebas)."""
        if self.existe():
            return False
        self.cliente.create_collection(
            collection_name=self.coleccion,
            vectors_config={self.vector: models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
                            "e5": models.VectorParams(size=dim_e5, distance=models.Distance.COSINE)},
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100))
        self.asegurar_indices()
        return True

    @staticmethod
    def id_punto(fuente_id: str, chunk_idx: int, chunk_size: int) -> str:
        # Mismo esquema determinista del ACA 1: recargar un documento sobrescribe en vez de duplicar
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{fuente_id}|{chunk_idx}|{chunk_size}"))

    def insertar(self, chunks: list[dict], vectores, lote: int = 64) -> int:
        puntos = [models.PointStruct(id=self.id_punto(c["fuente_id"], c["chunk_idx"], c["chunk_size"]),
                                     vector={self.vector: list(map(float, v))}, payload=c)
                  for c, v in zip(chunks, vectores)]
        for i in range(0, len(puntos), lote):
            self.cliente.upsert(self.coleccion, points=puntos[i:i + lote], wait=True)
        return len(puntos)

    def eliminar_fuente(self, fuente_id: str) -> int:
        if fuente_id in FUENTES_BASE:
            raise PermissionError("El corpus base del ACA 1 está protegido y no se puede eliminar desde la app.")
        filtro = self._filtro({"fuente_id": fuente_id})
        n = self.cliente.count(self.coleccion, count_filter=filtro, exact=True).count
        self.cliente.delete(self.coleccion, points_selector=models.FilterSelector(filter=filtro), wait=True)
        return n
