# ACA 2 · Computación en la Nube — Chatbot RAG de permanencia estudiantil

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BonnyGalindo77/aca2-rag-educacion-chatbot/blob/main/notebooks/ACA2_RAG_Chatbot_Educacion.ipynb)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aca2-rag-educacion.streamlit.app)

**CUN · Especialización en Inteligencia Artificial** · Docente: Hamilton Rivera Flor
**Equipo:** Bonny Darhyll Brayan Galindo Santos · Camilo José Rodríguez Ayubi · Andrés Alberto López Herrera
**Pilar:** 2 — Educación (consulta académica, normatividad institucional y permanencia estudiantil)

Chatbot RAG que responde preguntas sobre el **Reglamento Estudiantil de la CUN**, la **Guía de permanencia y graduación del MEN (2015)**,
los **Lineamientos de bienestar del MEN (2016)** y la evidencia del **dataset UCI de deserción**, citando documento y página.
Completa el ciclo iniciado en el [ACA 1](https://github.com/BonnyGalindo77/aca1-rag-educacion-qdrant): la base vectorial ya indexada en
Qdrant Cloud se conecta con un LLM en la nube (Google Gemini) y con una interfaz web en Streamlit.

## Arquitectura

```
Usuario ─► Streamlit (chat + gestión documental)
             │ pregunta + historial
             ▼
  Reescritura de seguimiento (Gemini, T = 0) ──► Embedding de consulta (gemini-embedding-2, 768 d)
             ▼
  Qdrant Cloud · colección aca1_educacion_permanencia · vector "gemini" · coseno · Top-k + filtros
             ▼
  Umbral de relevancia 0,645 ── sin evidencia ──► abstención (no llama al LLM)
             ▼
  Prompt aumentado (reglas + contexto numerado [n] + evidencia del dataset)
             ▼
  Gemini 3.5 Flash (T = 0,2 · top-p 0,9 · 2.048 tokens · streaming) ──► verificación de citas ──► respuesta + fuentes
```

| Capa | Herramienta | Notas |
|---|---|---|
| Base vectorial | Qdrant Cloud (free tier 1 GB) | Colección del ACA 1: 610 fragmentos, vectores nombrados `gemini` (768 d) y `e5` (384 d) |
| Embeddings | Gemini `gemini-embedding-2` | Mismo modelo y formato de entrada del ACA 1 (`task: search result \| query: …`) |
| LLM | Gemini API (`gemini-3.5-flash`, respaldo `gemini-flash-latest` / `-lite`) | Capa gratuita de Google AI Studio |
| Orquestación | Python nativo (`rag/`) | Sin LangChain en ejecución; `langchain-text-splitters` solo para ingesta |
| Interfaz | Streamlit | Desplegada en Streamlit Community Cloud |
| Analítica | scikit-learn | Modelo de alerta temprana de deserción (regresión logística) |

## Estructura del repositorio

```
├── app.py                         # interfaz Streamlit (chat, gestión documental, riesgo, evaluación)
├── rag/
│   ├── config.py                  # configuración y lectura segura de credenciales (st.secrets / .env)
│   ├── embeddings.py              # Gemini embeddings (consulta y documento), con reintentos y caché
│   ├── vectorstore.py             # Qdrant: búsqueda, inventario, fragmentos, carga y borrado protegido
│   ├── llm.py                     # cliente Gemini: hiperparámetros, streaming, reintentos y respaldo
│   ├── prompts.py                 # system prompt, plantillas de contexto, reescritura y juez
│   ├── pipeline.py                # canalización RAG con umbral, memoria y verificación de citas
│   ├── ingest.py                  # limpieza, chunking y etiquetado (idénticos al ACA 1)
│   ├── riesgo.py                  # predicción y explicación del modelo de alerta temprana
│   └── evaluacion.py              # Hit@k, MRR, LLM como juez, robustez y latencias
├── notebooks/ACA2_RAG_Chatbot_Educacion.ipynb   # notebook técnico reproducible (pasos 1 a 5)
├── scripts/
│   ├── preparar_datos.py          # capa SQL (SQLite) → fichas y evidencia por categoría
│   ├── entrenar_modelo_riesgo.py  # entrena y compara 3 modelos; guarda el mejor
│   └── indexar_corpus.py          # reconstruye la colección (solo si se pierde la del ACA 1)
├── data/                          # dataset UCI, corpus PDF, fichas y metadatos
├── models/modelo_riesgo.joblib
├── resultados/                    # métricas y figuras (las usa la app y el póster)
├── tests/test_offline.py          # pruebas sin red del flujo completo
├── .streamlit/config.toml · secrets.toml.example
├── .env.example · requirements.txt
```

## Ejecución local

```bash
git clone https://github.com/BonnyGalindo77/aca2-rag-educacion-chatbot.git
cd aca2-rag-educacion-chatbot
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                      # y pegue las llaves reales
streamlit run app.py
```

Las llaves son las mismas del ACA 1: `QDRANT_URL` y `QDRANT_API_KEY` (cloud.qdrant.io → clúster → API Keys) y `GOOGLE_API_KEY`
(aistudio.google.com/apikey).

## Despliegue en Streamlit Community Cloud

1. Suba este repositorio a GitHub (público). Verifique con `git status` que **no** aparezcan `.env` ni `.streamlit/secrets.toml`.
2. Entre a <https://share.streamlit.io> con su cuenta de GitHub → **Create app → Deploy a public app from GitHub**.
3. Repositorio `BonnyGalindo77/aca2-rag-educacion-chatbot`, rama `main`, archivo `app.py`. En **App URL** escriba `aca2-rag-educacion`.
4. **Advanced settings → Python 3.12** y en **Secrets** pegue el contenido de `.streamlit/secrets.toml.example` con los valores reales.
5. **Deploy**. La primera construcción tarda 2–4 minutos.

## Notebook

Ábralo con el botón de Colab, registre `QDRANT_URL`, `QDRANT_API_KEY` y `GOOGLE_API_KEY` en **🔑 Secretos** y ejecute
**Entorno de ejecución → Ejecutar todas** (≈ 5 minutos). Al final, `resultados/evaluacion_aca2.json` queda listo para la pestaña
*Evaluación* de la app.

## Pruebas sin conexión

```bash
python -m pytest -q tests          # Qdrant en memoria + vectorizador léxico + LLM extractivo
```

## Seguridad de credenciales

- Las llaves se leen solo de `st.secrets`, variables de entorno o `.env`; la interfaz y el notebook muestran si están cargadas, nunca su valor.
- `.env` y `.streamlit/secrets.toml` están en `.gitignore`.
- La carga y el borrado de documentos desde la app exigen `ADMIN_PASSWORD`; el corpus base del ACA 1 está protegido contra borrado.
- Si una llave se expone por error, revóquela en la consola de Qdrant o de AI Studio y genere otra.

## Corpus

| ID | Documento | Fragmentos |
|---|---|---|
| `reglamento_cun` | Reglamento Estudiantil CUN — Acuerdo 002 de 2011 (con modificaciones) | 113 |
| `guia_permanencia_men` | Guía para la implementación del modelo de gestión de permanencia y graduación estudiantil en IES (MEN, 2015) | 383 |
| `bienestar_men` | Lineamientos de política de bienestar para IES (MEN, 2016) | 104 |
| `dataset_uci` | Fichas analíticas del dataset UCI *Predict Students' Dropout and Academic Success* (4.424 estudiantes) | 10 |

La aplicación no necesita los PDF: consulta directamente los fragmentos ya indexados en Qdrant Cloud. Para reconstruir el índice,
`scripts/indexar_corpus.py` descarga cada PDF de su fuente oficial o de la copia del repositorio del ACA 1. El reglamento se obtiene del
Campus Virtual de la CUN; si el portal no responde, copie el PDF en `data/corpus/reglamento_estudiantil_cun.pdf`.

## Referencias principales (APA 7)

- Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS, 33*, 9459–9474.
- Ministerio de Educación Nacional. (2015). *Guía para la implementación del modelo de gestión de permanencia y graduación estudiantil en IES*.
- Ministerio de Educación Nacional. (2016). *Lineamientos de política de bienestar para instituciones de educación superior*.
- Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict students' dropout and academic success* [Conjunto de datos]. UCI. https://doi.org/10.24432/C5MC89
- Zheng, L., et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *NeurIPS, 36*.
