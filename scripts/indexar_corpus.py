"""(Re)construye la colección vectorial con el corpus base y las fichas del dataset.

Normalmente NO hace falta ejecutarlo: la aplicación usa la colección `aca1_educacion_permanencia` que
ya quedó indexada en Qdrant Cloud durante el ACA 1. Sirve para:
* recrear el índice si el clúster gratuito se elimina o se cambia de cuenta, y
* construir una copia local para pruebas sin conexión (EMBED_PROVIDER=hash, QDRANT_URL=local:./qdrant_local).

Uso:
    python scripts/indexar_corpus.py                 # usa la configuración del .env
    python scripts/indexar_corpus.py --local         # índice local de pruebas (sin red)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))


RESPALDO_ACA1 = "https://raw.githubusercontent.com/BonnyGalindo77/aca1-rag-educacion-qdrant/main/corpus/"


def descargar(doc, ruta):
    """Descarga el PDF de la fuente oficial o de la copia guardada en el repositorio del ACA 1."""
    import urllib.request

    for url in (doc["url"], RESPALDO_ACA1 + doc["url"].rsplit("/", 1)[-1]):
        try:
            datos = urllib.request.urlopen(url, timeout=60).read()
            if datos[:4] == b"%PDF":
                ruta.parent.mkdir(parents=True, exist_ok=True)
                ruta.write_bytes(datos)
                print(f"  descargado {ruta.name} desde {url.split('/')[2]}")
                return
        except Exception:
            continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true", help="índice local con vectorizador léxico de prueba")
    args = ap.parse_args()
    if args.local:
        os.environ.update({"QDRANT_URL": "local:" + str(RAIZ / "qdrant_local"), "EMBED_PROVIDER": "hash",
                           "LLM_PROVIDER": "offline"})

    from rag import Config, VectorStore
    from rag.embeddings import crear_embedder
    from rag.ingest import cargar_corpus_local, extraer_paginas, fragmentar

    cfg = Config.desde_entorno()
    store = VectorStore(cfg.qdrant_url, cfg.qdrant_api_key, cfg.coleccion, cfg.vector, cfg.embed_dim)
    print("Colección creada" if store.crear_coleccion() else "La colección ya existe; se actualizarán los puntos.")
    emb = crear_embedder(cfg)

    total = 0
    for doc in cargar_corpus_local():
        ruta = RAIZ / "data" / "corpus" / doc["archivo"]
        if not ruta.exists():
            descargar(doc, ruta)
        if not ruta.exists():
            print(f"  ⚠ falta {ruta.name}: se omite (ver README, sección corpus)")
            continue
        chunks = fragmentar(doc, extraer_paginas(ruta.read_bytes(), ruta.name), cfg.chunk_size, cfg.chunk_overlap,
                            emb.modelo)
        for c in chunks:
            c["origen"] = "corpus ACA 1"
        vec = emb.documentos([c["texto"] for c in chunks], [c["titulo"] for c in chunks])
        total += store.insertar(chunks, vec)
        print(f"  {doc['fuente_id']:<22} {len(chunks):>4} fragmentos")

    fichas = json.loads((RAIZ / "data" / "fichas_dataset.json").read_text(encoding="utf-8"))
    chunks = [{"fuente_id": "dataset_uci", "tipo_documento": "ficha_dataset", "titulo": f"Ficha analítica: {f['factor']}",
               "entidad": "UCI Machine Learning Repository", "anio": 2021, "url": "https://doi.org/10.24432/C5MC89",
               "pagina": 0, "chunk_idx": i, "texto": f["texto"], "factores_riesgo": [f["categoria"]],
               "variables_dataset": [f["variable"]], "evidencia_dataset": [f["evidencia"]],
               "n_caracteres": len(f["texto"]), "n_tokens_aprox": round(len(f["texto"]) / 4),
               "chunk_size": cfg.chunk_size, "pilar": "educacion", "modelo_principal": emb.modelo,
               "origen": "corpus ACA 1"} for i, f in enumerate(fichas)]
    total += store.insertar(chunks, emb.documentos([c["texto"] for c in chunks], [c["titulo"] for c in chunks]))
    print(f"  dataset_uci            {len(chunks):>4} fichas")
    print(f"Total insertado: {total} · puntos en la colección: {store.info()['puntos']}")


if __name__ == "__main__":
    main()
