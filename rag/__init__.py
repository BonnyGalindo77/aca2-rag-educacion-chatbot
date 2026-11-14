"""Paquete del chatbot RAG de permanencia estudiantil (ACA 2 · Computación en la Nube · CUN)."""
from .config import Config
from .embeddings import crear_embedder
from .llm import crear_llm
from .pipeline import RAGPipeline, Resultado, analizar_citas
from .vectorstore import VectorStore


def crear_sistema(config: Config | None = None, modelo_llm: str | None = None) -> RAGPipeline:
    config = config or Config.desde_entorno()
    store = VectorStore(config.qdrant_url, config.qdrant_api_key, config.coleccion, config.vector, config.embed_dim)
    return RAGPipeline(store, crear_embedder(config), crear_llm(config, modelo_llm), config)


__all__ = ["Config", "RAGPipeline", "Resultado", "VectorStore", "crear_sistema", "analizar_citas"]
