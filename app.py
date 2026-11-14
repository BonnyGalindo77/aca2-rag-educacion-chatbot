"""Permanencia CUN · Chatbot RAG sobre Qdrant Cloud + Gemini.

ACA 2 · Computación en la Nube · Especialización en Inteligencia Artificial · CUN
Ejecutar:  streamlit run app.py
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from rag import Config, crear_sistema
from rag.ingest import extraer_paginas, fragmentar, slug
from rag.llm import GeminiLLM
from rag.vectorstore import FUENTES_BASE

RAIZ = Path(__file__).resolve().parent
EQUIPO = ["Bonny Darhyll Brayan Galindo Santos", "Camilo José Rodríguez Ayubi", "Andrés Alberto López Herrera"]
ENLACES = {
    "Repositorio ACA 2": "https://github.com/BonnyGalindo77/aca2-rag-educacion-chatbot",
    "Repositorio ACA 1": "https://github.com/BonnyGalindo77/aca1-rag-educacion-qdrant",
}
ETIQUETAS_TIPO = {"reglamento": "Reglamento CUN", "guia_permanencia": "Guía permanencia MEN",
                  "politica_bienestar": "Bienestar MEN", "ficha_dataset": "Ficha dataset UCI"}
ETIQUETAS_RIESGO = {"financiero": "Financiero", "academico": "Académico",
                    "socioeconomico_personal": "Socioeconómico / personal",
                    "bienestar_institucional": "Bienestar institucional", "normativo": "Normativo"}
COLORES = {"reglamento_cun": "#0B4F6C", "guia_permanencia_men": "#E07A1F", "bienestar_men": "#2E8B57",
           "dataset_uci": "#7B4FA0"}
SUGERENCIAS = [
    "¿Qué pasa si pierdo la misma asignatura por tercera vez?",
    "Tengo deudas con la universidad y pienso retirarme, ¿qué opciones tengo?",
    "¿Qué estrategias recomienda el MEN para estudiantes con dificultades económicas?",
    "¿Qué tanto influye aprobar pocas materias en primer semestre en la deserción?",
]

st.set_page_config(page_title="Permanencia CUN · Chatbot RAG", page_icon=":material/school:", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
.hero {background: linear-gradient(120deg,#0B3C5D 0%,#0B4F6C 60%,#1F7A8C 100%); color:#fff; border-radius:14px;
       padding:18px 24px; margin-bottom:12px;}
.hero h1 {color:#fff; font-size:1.65rem; margin:0 0 4px 0;}
.hero p {color:#E6F1F5; margin:0; font-size:.95rem;}
.hero .acento {color:#FFB347; font-weight:600;}
.fuente {border-left:4px solid #0B4F6C; background:rgba(11,79,108,.06); border-radius:6px; padding:8px 12px; margin:6px 0;}
.fuente .cab {font-weight:600; font-size:.9rem;}
.fuente .meta {font-size:.78rem; opacity:.75;}
.chip {display:inline-block; padding:1px 8px; border-radius:10px; font-size:.72rem; margin-right:4px;
       background:#E8F1F5; color:#0B3C5D; border:1px solid #C7DCE6;}
.kpi-nota {font-size:.78rem; opacity:.7;}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ recursos con caché ------
@st.cache_resource(show_spinner="Conectando con Qdrant Cloud y Gemini…")
def obtener_sistema(modelo: str):
    return crear_sistema(Config.desde_entorno(), modelo_llm=modelo)


@st.cache_resource
def obtener_modelo_riesgo():
    from rag.riesgo import ModeloRiesgo
    return ModeloRiesgo()


@st.cache_data(ttl=3600, show_spinner=False)
def modelos_llm(_cfg_key: str) -> list[str]:
    cfg = Config.desde_entorno()
    if cfg.proveedor_llm != "gemini" or not cfg.google_api_key:
        return [cfg.llm_model]
    try:
        disponibles = GeminiLLM(cfg.google_api_key, cfg.llm_model).modelos_disponibles()
    except Exception:
        disponibles = []
    preferidos = [cfg.llm_model, "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-flash-latest"]
    orden = [m for m in preferidos if not disponibles or m in disponibles or m == cfg.llm_model]
    return list(dict.fromkeys(orden + [m for m in disponibles if "flash" in m]))


@st.cache_data(ttl=300, show_spinner=False)
def info_indice(_store, version: int):
    t = _store.ping()
    return {**_store.info(), "ping_ms": t}


@st.cache_data(ttl=300, show_spinner=False)
def fuentes_indice(_store, version: int):
    return _store.fuentes()


@st.cache_data(ttl=300, show_spinner=False)
def fragmentos_fuente(_store, fuente_id: str, version: int):
    return _store.fragmentos(fuente_id)


def cargar_json(ruta: Path):
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


# ------------------------------------------------------------------ estado de sesión --------
ss = st.session_state
ss.setdefault("mensajes", [])
ss.setdefault("version_indice", 0)
ss.setdefault("pendiente", None)
cfg = Config.desde_entorno()

# ------------------------------------------------------------------ barra lateral -----------
with st.sidebar:
    logo = RAIZ / "assets" / "logo_cun.png"
    if logo.exists():
        st.image(str(logo), width=170)
    st.markdown("### Permanencia CUN")
    st.caption("Chatbot RAG de consulta académica y permanencia estudiantil")

    with st.expander(":material/tune: Configuración del modelo", expanded=False):
        lista = modelos_llm(cfg.llm_model)
        modelo = st.selectbox("Modelo LLM (Gemini API)", lista, index=0,
                              help="Se listan los modelos de texto habilitados para la API key.")
        temperatura = st.slider("Temperatura", 0.0, 1.0, cfg.temperatura, 0.05,
                                help="Baja = respuestas estables y apegadas a los documentos.")
        top_p = st.slider("Top-p", 0.1, 1.0, cfg.top_p, 0.05)
        max_tokens = st.select_slider("Máx. tokens de salida", [256, 512, 768, 1024, 1536, 2048], cfg.max_tokens)
        st.divider()
        top_k = st.slider("Fragmentos recuperados (Top-k)", 1, 10, cfg.top_k)
        umbral = st.slider("Umbral de relevancia (coseno)", 0.40, 0.90, cfg.umbral, 0.005,
                           help="Calibrado en el ACA 1: 0,645. Por debajo, el asistente se abstiene.")
        reescribir = st.toggle("Reescribir preguntas de seguimiento", True,
                               help="Usa el historial para volver autónoma una pregunta como '¿y el plazo?'.")
        streaming = st.toggle("Respuesta en streaming", True)

    with st.expander(":material/filter_alt: Filtros de búsqueda", expanded=False):
        f_tipo = st.multiselect("Tipo de documento", list(ETIQUETAS_TIPO), format_func=lambda x: ETIQUETAS_TIPO.get(x, x))
        f_riesgo = st.multiselect("Categoría de riesgo", list(ETIQUETAS_RIESGO),
                                  format_func=lambda x: ETIQUETAS_RIESGO.get(x, x))
    filtros = {"tipo_documento": f_tipo, "factores_riesgo": f_riesgo}

    c1, c2 = st.columns(2)
    if c1.button(":material/add_comment: Nuevo chat", width="stretch"):
        ss.mensajes = []
        st.rerun()
    if ss.mensajes:
        md = "\n\n".join(f"**{'Tú' if m['role'] == 'user' else 'Asistente'}:** {m['content']}" +
                         ("".join(f"\n- [{f['n']}] {f['cita']}" for f in m.get("fuentes", [])) if m["role"] == "assistant" else "")
                         for m in ss.mensajes)
        c2.download_button(":material/download: Exportar", md, file_name=f"chat_permanencia_{datetime.now():%Y%m%d_%H%M}.md",
                           width="stretch")

    st.markdown("##### Estado de los servicios")
    estado = cfg.estado_credenciales()
    st.markdown(f"{'🟢' if estado['QDRANT_URL'] and estado['QDRANT_API_KEY'] else '🟠'} Qdrant Cloud &nbsp; "
                f"{'🟢' if estado['GOOGLE_API_KEY'] else '🔴'} Gemini API")
    st.caption("Las llaves se leen de `st.secrets` / `.env` y nunca se muestran.")
    st.divider()
    st.caption("**Equipo:** " + " · ".join(EQUIPO))
    st.caption("Docente: Hamilton Rivera Flor · CUN · Especialización en IA · 2026")

# ------------------------------------------------------------------ encabezado --------------
st.markdown("""<div class="hero"><h1>Asistente de permanencia estudiantil</h1>
<p>Pregunta en lenguaje natural sobre el <b>Reglamento Estudiantil CUN</b>, las guías de <b>permanencia y bienestar
del MEN</b> y la evidencia del <b>dataset UCI de deserción</b>. Cada respuesta cita el documento y la página.
<span class="acento">RAG · Qdrant Cloud · Gemini</span></p></div>""", unsafe_allow_html=True)

faltan = [k for k in ("GOOGLE_API_KEY", "QDRANT_URL") if not estado[k]]
if faltan and cfg.proveedor_llm == "gemini":
    st.error(f"Faltan credenciales: {', '.join(faltan)}. Configure `.streamlit/secrets.toml` o el archivo `.env` "
             "(vea `secrets.toml.example`) y recargue la página.", icon=":material/key_off:")
    st.stop()

try:
    sistema = obtener_sistema(modelo)
except Exception as e:
    st.error(f"No fue posible iniciar el sistema: {e}", icon=":material/error:")
    st.stop()
store = sistema.store


# ------------------------------------------------------------------ utilidades de UI --------
def segundos(ms):
    return f"{(ms or 0) / 1000:.2f} s".replace(".", ",")


def mostrar_fuentes(fuentes, descartadas=None, citadas=None):
    if not fuentes and not descartadas:
        return
    citadas = set(citadas or [])
    if fuentes:
        titulo = f"Fuentes recuperadas ({len(fuentes)})" + (f" · citadas: {', '.join(map(str, sorted(citadas)))}" if citadas else "")
    else:
        titulo = f"Ningún fragmento superó el umbral ({len(descartadas)} descartados)"
    with st.expander(f":material/menu_book: {titulo}", expanded=False):
        for f in fuentes:
            pag = f"p. {f['pagina']}" if f.get("pagina") else "ficha analítica"
            chips = "".join(f"<span class='chip'>{ETIQUETAS_RIESGO.get(x, x)}</span>" for x in f.get("factores_riesgo", []))
            marca = "✅ citada" if f["n"] in citadas else "recuperada"
            st.markdown(
                f"<div class='fuente' style='border-left-color:{COLORES.get(f['fuente_id'], '#888')}'>"
                f"<div class='cab'>[{f['n']}] {f['titulo']}</div>"
                f"<div class='meta'>{f.get('entidad', '')} · {f.get('anio', '')} · {pag} · similitud {f['score']:.3f} · {marca}</div>"
                f"<div>{chips}</div></div>", unsafe_allow_html=True)
            texto = f["texto"].strip()
            st.caption(texto[:700] + ("…" if len(texto) > 700 else ""))
            if f.get("evidencia_dataset"):
                st.caption("📊 Evidencia del dataset asociada: " + "; ".join(f["evidencia_dataset"]))
        if descartadas:
            st.caption("Fragmentos por debajo del umbral (no enviados al LLM): " + ", ".join(
                f"{d['fuente_id']} p.{d['pagina']} ({d['score']:.3f})" for d in descartadas))


def linea_metricas(meta):
    t = meta.get("tiempos", {})
    partes = [f"⏱ total {segundos(t.get('total_ms'))}",
              f"búsqueda {segundos((t.get('embedding_ms') or 0) + (t.get('qdrant_ms') or 0))}"]
    if t.get("llm_ms"):
        partes.append(f"LLM {segundos(t['llm_ms'])}")
    if meta.get("modelo"):
        partes.append(meta["modelo"])
    if meta.get("uso"):
        partes.append(f"{meta['uso'].get('tokens_entrada', 0)}→{meta['uso'].get('tokens_salida', 0)} tokens")
    if meta.get("citas") and meta.get("estado") == "ok":
        partes.append(f"cobertura de citas {meta['citas'].get('cobertura', 0):.0%}")
    st.caption(" · ".join(partes))
    if meta.get("consulta") and meta["consulta"] != meta.get("pregunta"):
        st.caption(f"🔎 Consulta interpretada con el historial: *{meta['consulta']}*")


def ejecutar_rag(pregunta, historial, contenedor, stream=True):
    with contenedor:
        with st.spinner("Buscando en la base vectorial de Qdrant Cloud…"):
            r = sistema.responder(pregunta, historial=historial, k=top_k, umbral=umbral, filtros=filtros,
                                  reescribir=reescribir, stream=stream, temperatura=temperatura, top_p=top_p,
                                  max_tokens=max_tokens)
        if r.flujo is not None:
            try:
                st.write_stream(r.flujo)
            except Exception as e:
                r.estado, r.error = "error_llm", str(e)[:300]
                r.respuesta = r.respuesta or "La generación se interrumpió. Intenta de nuevo en unos segundos."
                st.warning(r.respuesta)
        elif r.estado == "sin_evidencia":
            st.info(r.respuesta, icon=":material/search_off:")
        elif r.estado == "error_llm":
            st.warning(r.respuesta, icon=":material/cloud_off:")
            with st.expander("Detalle técnico"):
                st.code(r.error)
        else:
            st.markdown(r.respuesta)
        meta = {"pregunta": r.pregunta, "consulta": r.consulta, "estado": r.estado, "tiempos": r.tiempos,
                "modelo": r.modelo, "uso": r.uso, "citas": r.citas, "fuentes": r.fuentes, "descartadas": r.descartadas}
        mostrar_fuentes(r.fuentes, r.descartadas, r.citas.get("citadas"))
        linea_metricas(meta)
    return r, meta


# ------------------------------------------------------------------ pestañas ----------------
tab_chat, tab_docs, tab_riesgo, tab_eval, tab_info = st.tabs([
    ":material/forum: Chatbot", ":material/folder_open: Gestión documental",
    ":material/monitoring: Riesgo de deserción", ":material/fact_check: Evaluación",
    ":material/info: Acerca del proyecto"])

# ======================================================================= CHAT ==============
with tab_chat:
    if not ss.mensajes:
        st.markdown("**Prueba con una de estas preguntas** o escribe la tuya abajo:")
        cols = st.columns(len(SUGERENCIAS))
        for c, s in zip(cols, SUGERENCIAS):
            if c.button(s, width="stretch"):
                ss.pendiente = s
                st.rerun()

    for i, m in enumerate(ss.mensajes):
        with st.chat_message(m["role"], avatar=":material/person:" if m["role"] == "user" else ":material/school:"):
            if m["role"] == "assistant" and m.get("estado") == "sin_evidencia":
                st.info(m["content"], icon=":material/search_off:")
            else:
                st.markdown(m["content"])
            if m["role"] == "assistant":
                mostrar_fuentes(m.get("fuentes", []), m.get("descartadas"), m.get("citas", {}).get("citadas"))
                linea_metricas(m)
                st.feedback("thumbs", key=f"fb_{i}")

    pregunta = st.chat_input("Escribe tu pregunta sobre reglamento, permanencia o bienestar…")
    pregunta = pregunta or ss.pop("pendiente", None)
    if pregunta:
        historial = [{"role": m["role"], "content": m["content"]} for m in ss.mensajes]
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(pregunta)
        ss.mensajes.append({"role": "user", "content": pregunta})
        r, meta = ejecutar_rag(pregunta, historial, st.chat_message("assistant", avatar=":material/school:"),
                               stream=streaming)
        ss.mensajes.append({"role": "assistant", "content": r.respuesta, **meta})
        st.rerun()

# ======================================================================= GESTIÓN DOCUMENTAL ==
with tab_docs:
    try:
        info = info_indice(store, ss.version_indice)
        fuentes = fuentes_indice(store, ss.version_indice)
    except Exception as e:
        st.error(f"No se pudo leer la colección `{cfg.coleccion}`: {e}")
        info, fuentes = None, []

    if info:
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Fragmentos indexados", f"{info['puntos']:,}".replace(",", "."))
        k2.metric("Documentos / fuentes", len(fuentes))
        k3.metric("Dimensión del vector", f"{info['vectores'].get(cfg.vector, {}).get('dim', '?')} d")
        k4.metric("Estado del clúster", info["estado"].upper())
        k5.metric("Latencia Qdrant", f"{info['ping_ms']:.0f} ms")
        st.caption(f"Colección `{cfg.coleccion}` · {store.modo} · vectores nombrados: "
                   + ", ".join(f"`{n}` ({v['dim']} d, {v['distancia']})" for n, v in info["vectores"].items())
                   + f" · índices de payload: {', '.join(info['indices_payload']) or '—'}"
                   + f" · HNSW m={info['hnsw']['m']}, ef_construct={info['hnsw']['ef_construct']}")

    if fuentes:
        df_f = pd.DataFrame(fuentes)
        izq, der = st.columns([3, 2])
        with izq:
            st.markdown("##### Documentos en la base vectorial")
            st.dataframe(
                df_f[["titulo", "anio", "tipo_documento", "paginas", "fragmentos"]],
                hide_index=True, width="stretch",
                column_config={
                    "titulo": st.column_config.TextColumn("Documento", width="medium"),
                    "anio": st.column_config.NumberColumn("Año", format="%d"),
                    "tipo_documento": "Tipo", "paginas": st.column_config.NumberColumn("Págs.", format="%d"),
                    "fragmentos": st.column_config.ProgressColumn("Fragmentos", format="%d", min_value=0,
                                                                  max_value=int(df_f.fragmentos.max()))})
        with der:
            st.markdown("##### Cobertura por categoría de riesgo")
            filas = [{"fuente": f["fuente_id"], "categoria": ETIQUETAS_RIESGO.get(c, c), "fragmentos": n}
                     for f in fuentes for c, n in f["factores_riesgo"].items()]
            if filas:
                st.altair_chart(alt.Chart(pd.DataFrame(filas)).mark_bar().encode(
                    x=alt.X("sum(fragmentos):Q", title="fragmentos etiquetados"),
                    y=alt.Y("categoria:N", sort="-x", title=None),
                    color=alt.Color("fuente:N", scale=alt.Scale(domain=list(COLORES), range=list(COLORES.values())),
                                    legend=alt.Legend(orient="bottom", title=None, columns=2, labelLimit=220)),
                    tooltip=["fuente", "categoria", "fragmentos"]).properties(height=230), width="stretch")

        st.markdown("##### Explorar fragmentos y metadatos")
        e1, e2 = st.columns([2, 3])
        fuente_sel = e1.selectbox("Documento", df_f.fuente_id, format_func=lambda x: df_f.set_index("fuente_id").titulo[x])
        texto_filtro = e2.text_input("Filtrar por texto dentro del documento", placeholder="Ej.: reintegro, artículo 35, beca…")
        frag = pd.DataFrame(fragmentos_fuente(store, fuente_sel, ss.version_indice))
        if texto_filtro:
            frag = frag[frag.texto.str.contains(texto_filtro, case=False, na=False)]
        if not frag.empty:
            vista = frag.assign(factores=frag.factores_riesgo.apply(lambda x: ", ".join(x or [])),
                                vista_previa=frag.texto.str.replace("\n", " ").str.slice(0, 160) + "…")
            sel = st.dataframe(vista[["pagina", "chunk_idx", "n_caracteres", "factores", "vista_previa"]],
                               hide_index=True, width="stretch", height=260, on_select="rerun",
                               selection_mode="single-row",
                               column_config={"pagina": "Pág.", "chunk_idx": "Chunk", "n_caracteres": "Caracteres",
                                              "factores": "Categorías de riesgo", "vista_previa": "Texto"})
            filas_sel = sel.selection.rows if sel and sel.selection else []
            if filas_sel:
                c = vista.iloc[filas_sel[0]]
                with st.container(border=True):
                    st.markdown(f"**{c.titulo}** · pág. {c.pagina} · chunk {c.chunk_idx} · `{c.id}`")
                    st.text(c.texto)
                    if c.get("evidencia_dataset"):
                        st.caption("📊 " + "; ".join(c.evidencia_dataset))
            else:
                st.caption(f"{len(frag)} fragmentos. Selecciona una fila para ver el texto completo y sus metadatos.")
        else:
            st.caption("Sin coincidencias.")

    st.markdown("##### Prueba de búsqueda semántica (solo recuperación)")
    b1, b2 = st.columns([4, 1])
    consulta_prueba = b1.text_input("Consulta", placeholder="¿Cuál es el plazo para solicitar reintegro?",
                                    label_visibility="collapsed")
    if b2.button(":material/search: Buscar", width="stretch") and consulta_prueba:
        puntos, _, t = sistema.recuperar(consulta_prueba, k=max(top_k, 8), umbral=umbral, filtros=filtros)
        res = pd.DataFrame([{"rank": i + 1, "score": round(p.score, 4), "fuente": p.payload.get("fuente_id"),
                             "pagina": p.payload.get("pagina"),
                             "fragmento": p.payload.get("texto", "")[:220].replace("\n", " ") + "…"}
                            for i, p in enumerate(puntos)])
        st.caption(f"Embedding Gemini {t['embedding_ms']:.0f} ms + búsqueda Qdrant {t['qdrant_ms']:.0f} ms · "
                   f"umbral {umbral:.3f}")
        if not res.empty:
            res["etiqueta"] = res["rank"].astype(str) + ". " + res.fuente + " p." + res.pagina.astype(str)
            barras = alt.Chart(res).mark_bar(clip=True).encode(
                x=alt.X("score:Q", scale=alt.Scale(domain=[max(0, res.score.min() - .05), min(1, res.score.max() + .03)]),
                        title="similitud coseno"),
                y=alt.Y("etiqueta:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
                color=alt.Color("fuente:N", scale=alt.Scale(domain=list(COLORES), range=list(COLORES.values())), legend=None),
                tooltip=["rank", "fuente", "pagina", "score"])
            regla = alt.Chart(pd.DataFrame({"u": [umbral]})).mark_rule(color="#C0392B", strokeDash=[4, 3]).encode(x="u:Q")
            st.altair_chart((barras + regla).properties(height=34 * len(res)), width="stretch")
            st.dataframe(res.drop(columns="etiqueta"), hide_index=True, width="stretch")

    with st.expander(":material/admin_panel_settings: Cargar o retirar documentos (administrador)"):
        if not cfg.admin_password:
            st.info("La carga está deshabilitada: defina `ADMIN_PASSWORD` en los secretos para habilitarla. "
                    "Así nadie externo puede escribir en la base vectorial pública.")
        else:
            clave = st.text_input("Clave de administrador", type="password")
            if clave and clave != cfg.admin_password:
                st.error("Clave incorrecta.")
            elif clave == cfg.admin_password:
                with st.form("carga", clear_on_submit=True):
                    archivo = st.file_uploader("Documento (PDF, TXT o MD)", type=["pdf", "txt", "md"])
                    a1, a2 = st.columns(2)
                    titulo = a1.text_input("Título")
                    entidad = a2.text_input("Entidad", value="CUN")
                    a3, a4 = st.columns(2)
                    anio = a3.number_input("Año", 1990, 2030, datetime.now().year)
                    tipo = a4.selectbox("Tipo", ["reglamento", "guia_permanencia", "politica_bienestar", "acuerdo",
                                                 "circular", "documento_cargado"])
                    enviar = st.form_submit_button(":material/upload: Procesar e indexar")
                if enviar and archivo and titulo:
                    with st.status("Indexando documento…", expanded=True) as s:
                        st.write("1/4 Extrayendo y limpiando texto…")
                        paginas = extraer_paginas(archivo.getvalue(), archivo.name)
                        meta_doc = {"fuente_id": "doc_" + slug(titulo), "titulo": titulo, "entidad": entidad,
                                    "anio": int(anio), "tipo_documento": tipo, "url": ""}
                        st.write(f"2/4 Fragmentando {len(paginas)} páginas (1.500 car., 15 % solapamiento)…")
                        chunks = fragmentar(meta_doc, paginas, cfg.chunk_size, cfg.chunk_overlap, cfg.embed_model)
                        st.write(f"3/4 Generando {len(chunks)} embeddings con {cfg.embed_model}…")
                        barra = st.progress(0.0)
                        vec = sistema.embedder.documentos([c["texto"] for c in chunks], [titulo] * len(chunks),
                                                          progreso=barra.progress)
                        st.write("4/4 Insertando en Qdrant Cloud (upsert por lotes)…")
                        store.asegurar_indices()
                        n = store.insertar(chunks, vec)
                        s.update(label=f"Documento indexado: {n} fragmentos", state="complete")
                    ss.version_indice += 1
                elif enviar:
                    st.warning("Adjunte un archivo y escriba un título.")

                retirables = [f for f in fuentes if f["fuente_id"] not in FUENTES_BASE]
                if retirables:
                    r1, r2 = st.columns([3, 1])
                    quitar = r1.selectbox("Retirar documento cargado", [f["fuente_id"] for f in retirables])
                    if r2.button("Retirar", type="secondary") and quitar:
                        n = store.eliminar_fuente(quitar)
                        ss.version_indice += 1
                        st.success(f"Se eliminaron {n} fragmentos de `{quitar}`.")
                st.caption("El corpus base del ACA 1 (reglamento, guías MEN y fichas UCI) está protegido contra borrado.")

# ======================================================================= RIESGO ============
with tab_riesgo:
    try:
        mr = obtener_modelo_riesgo()
    except Exception as e:
        mr = None
        st.error(f"No se pudo cargar el modelo de riesgo: {e}")
    if mr:
        met = mr.metricas.get("prueba_20pct", {})
        st.markdown("Modelo de **alerta temprana** entrenado con el dataset UCI (4.424 estudiantes) con variables "
                    "disponibles al cierre del primer semestre. El resultado se conecta con el chatbot para sugerir "
                    "normas y estrategias de apoyo.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ROC-AUC (prueba)", f"{met.get('roc_auc', 0):.3f}")
        m2.metric("PR-AUC", f"{met.get('pr_auc', 0):.3f}")
        m3.metric("Recall deserción", f"{met.get('recall', 0):.1%}")
        m4.metric("F1", f"{met.get('f1', 0):.3f}")
        st.caption(f"Modelo seleccionado por validación cruzada (5 pliegues): **{mr.nombre}** · conjunto de prueba "
                   f"independiente del 20 % · umbral de decisión 0,5.")

        with st.form("riesgo"):
            c1, c2, c3 = st.columns(3)
            edad = c1.number_input("Edad al ingresar", 17, 70, 19)
            sexo = c1.radio("Sexo", ["Mujer", "Hombre"], horizontal=True)
            jornada = c1.radio("Jornada", ["Diurna", "Nocturna"], horizontal=True)
            admision = c1.slider("Puntaje de admisión (95–190)", 95.0, 190.0, 127.0, 0.5)
            becado = c2.toggle("Tiene beca", False)
            deudor = c2.toggle("Tiene deudas con la institución", False)
            al_dia = c2.toggle("Matrícula al día", True)
            desplazado = c2.toggle("Vive desplazado de su lugar de origen", False)
            inscritas = c3.number_input("Asignaturas inscritas sem. 1", 0, 26, 6)
            evaluaciones = c3.number_input("Evaluaciones presentadas sem. 1", 0, 45, 8)
            aprobadas = c3.number_input("Asignaturas aprobadas sem. 1", 0, 26, 5)
            promedio = c3.slider("Promedio sem. 1 (escala 0–20)", 0.0, 20.0, 12.5, 0.1)
            calcular = st.form_submit_button(":material/calculate: Estimar riesgo", type="primary")

        if calcular:
            ss.valores_riesgo = {
                "Age at enrollment": edad, "Gender": 1 if sexo == "Hombre" else 0, "Scholarship holder": int(becado),
                "Debtor": int(deudor), "Tuition fees up to date": int(al_dia), "Displaced": int(desplazado),
                "Daytime/evening attendance": 1 if jornada == "Diurna" else 0, "Admission grade": admision,
                "Curricular units 1st sem (enrolled)": inscritas, "Curricular units 1st sem (evaluations)": evaluaciones,
                "Curricular units 1st sem (approved)": min(aprobadas, inscritas),
                "Curricular units 1st sem (grade)": promedio}
            ss.resultado_riesgo = mr.predecir(ss.valores_riesgo)
            ss.pop("orientacion", None)

        if ss.get("resultado_riesgo"):
            res = ss.resultado_riesgo
            colores = {"bajo": "green", "medio": "orange", "alto": "red"}
            r1, r2 = st.columns([1, 2])
            with r1:
                st.metric("Probabilidad estimada de deserción", f"{res['probabilidad']:.1%}",
                          delta=f"{(res['probabilidad'] - 0.321) * 100:+.1f} pp vs. media de la cohorte (32,1 %)",
                          delta_color="inverse")
                st.markdown(f"Nivel de riesgo: :{colores[res['banda']]}-badge[{res['banda'].upper()}]")
                st.progress(res["probabilidad"])
                st.caption("Herramienta de apoyo académico, no un diagnóstico. Datos de una institución portuguesa.")
            with r2:
                contrib = pd.DataFrame(res["contribuciones"][:8])
                if not contrib.empty:
                    contrib["efecto"] = contrib.aporte.apply(lambda a: "aumenta el riesgo" if a > 0 else "reduce el riesgo")
                    st.altair_chart(alt.Chart(contrib).mark_bar().encode(
                        x=alt.X("aporte:Q", title="aporte al logit (regresión logística)"),
                        y=alt.Y("variable:N", sort=alt.EncodingSortField("aporte", op="sum", order="descending"), title=None,
                                axis=alt.Axis(labelLimit=240, labelOverlap=False)),
                        color=alt.Color("efecto:N", scale=alt.Scale(domain=["aumenta el riesgo", "reduce el riesgo"],
                                                                     range=["#C0392B", "#2E8B57"]),
                                        legend=alt.Legend(orient="bottom", title=None)),
                        tooltip=["variable", alt.Tooltip("aporte:Q", format=".2f")]).properties(
                        height=42 * len(contrib), title="Factores que más pesan en esta estimación"), width="stretch")

            from rag.riesgo import pregunta_orientacion
            consulta_or = pregunta_orientacion(ss.valores_riesgo, res)
            st.markdown("##### Orientación basada en documentos")
            st.caption(f"Consulta generada para el RAG: *{consulta_or}*")
            if st.button(":material/auto_awesome: Pedir orientación al asistente"):
                r, meta = ejecutar_rag(consulta_or, [], st.container(border=True), stream=False)
                ss.orientacion = {"respuesta": r.respuesta, **meta}
            elif ss.get("orientacion"):
                with st.container(border=True):
                    st.markdown(ss.orientacion["respuesta"])
                    mostrar_fuentes(ss.orientacion["fuentes"], None, ss.orientacion["citas"].get("citadas"))

# ======================================================================= EVALUACIÓN ========
with tab_eval:
    ev = cargar_json(RAIZ / "resultados" / "evaluacion_aca2.json")
    st.markdown("Resultados del **Paso 5 (validación integral)**, generados con el notebook "
                "`notebooks/ACA2_RAG_Chatbot_Educacion.ipynb` sobre Qdrant Cloud y Gemini.")
    if ev:
        rec, gen = ev.get("recuperacion", {}), ev.get("generacion", {})
        st.markdown("##### Recuperación (14 consultas de validación del ACA 1)")
        a = st.columns(5)
        a[0].metric("Hit@1", f"{rec.get('Hit@1', 0):.2f}")
        a[1].metric("Hit@3", f"{rec.get('Hit@3', 0):.2f}")
        a[2].metric("MRR", f"{rec.get('MRR', 0):.3f}")
        a[3].metric("Embedding (mediana)", f"{rec.get('lat_embedding_ms_mediana', 0):.0f} ms")
        a[4].metric("Qdrant (mediana)", f"{rec.get('lat_qdrant_ms_mediana', 0):.0f} ms")
        st.markdown("##### Generación y robustez")
        b = st.columns(5)
        b[0].metric("Comportamiento correcto", f"{gen.get('comportamiento_correcto', 0):.0%}")
        b[1].metric("Fidelidad (juez 1–5)", f"{gen.get('juez_fidelidad_media', 0):.2f}")
        b[2].metric("Relevancia (juez 1–5)", f"{gen.get('juez_relevancia_media', 0):.2f}")
        b[3].metric("Abstención fuera de dominio", f"{gen.get('abstencion_correcta_fuera_dominio', 0):.0%}")
        b[4].metric("Latencia total (mediana)", segundos(gen.get("latencia_total_ms_mediana")))
        casos = pd.DataFrame(ev.get("casos", []))
        if not casos.empty:
            casos["fidelidad"] = casos.get("juez", pd.Series([{}] * len(casos))).apply(
                lambda j: j.get("fidelidad") if isinstance(j, dict) else None)
            st.dataframe(casos[["id", "tipo", "pregunta", "estado", "comportamiento_correcto", "fidelidad",
                                "cobertura_citas", "latencia_total_ms"]], hide_index=True, width="stretch",
                         column_config={"comportamiento_correcto": st.column_config.CheckboxColumn("Correcto"),
                                        "cobertura_citas": st.column_config.NumberColumn("Cobertura citas", format="%.2f")})
            with st.expander("Ver respuestas completas"):
                for c in ev.get("casos", []):
                    st.markdown(f"**{c['id']} · {c['tipo']}** — {c['pregunta']}")
                    st.markdown(c["respuesta"])
                    st.divider()
        st.caption(f"Corrida: {ev.get('fecha', '')} · modelo {ev.get('modelo_llm', '')}")
    else:
        st.info("Aún no hay resultados guardados. Ejecute el notebook o use el botón de abajo.")

    st.markdown("##### Ejecutar ahora la evaluación de recuperación")
    if st.button(":material/play_arrow: Evaluar recuperación (14 consultas)"):
        from rag.evaluacion import evaluar_recuperacion
        with st.spinner("Consultando Qdrant Cloud…"):
            m, filas = evaluar_recuperacion(sistema)
        st.json(m)
        st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")

# ======================================================================= ACERCA ============
with tab_info:
    arq = RAIZ / "assets" / "arquitectura_rag.png"
    if arq.exists():
        st.image(str(arq), caption="Arquitectura RAG end-to-end", width="stretch")
    i1, i2 = st.columns([3, 2])
    with i1:
        st.markdown("""
**Pregunta orientadora.** ¿Cómo articular un modelo de lenguaje en la nube con una base vectorial serverless para
construir un chatbot RAG confiable, con una interfaz que facilite la administración documental, la consulta en lenguaje
natural y la reducción de alucinaciones?

**Cómo funciona.** Tu pregunta se convierte en un vector con `gemini-embedding-2` (768 d), Qdrant Cloud devuelve los
fragmentos más parecidos, solo pasan los que superan el umbral calibrado (0,645) y Gemini redacta la respuesta usando
exclusivamente esos fragmentos, citándolos con [n]. Si no hay evidencia, el asistente se abstiene en lugar de inventar.
""")
    with i2:
        st.markdown("##### Stack (capa gratuita)")
        st.dataframe(pd.DataFrame([
            ("Base vectorial", "Qdrant Cloud · colección del ACA 1 · coseno"),
            ("Embeddings", "Gemini gemini-embedding-2 · 768 d"),
            ("LLM", f"Gemini API · {modelo}"),
            ("Orquestación", "Python nativo (sin LangChain en el runtime)"),
            ("Interfaz", "Streamlit · Community Cloud"),
            ("Analítica", "scikit-learn · regresión logística"),
            ("Credenciales", "st.secrets / .env (nunca en el repositorio)"),
        ], columns=["Capa", "Herramienta"]), hide_index=True, width="stretch")
        st.markdown("##### Equipo")
        st.markdown("\n".join(f"- {n}" for n in EQUIPO))
        st.markdown("Docente: **Hamilton Rivera Flor**")
        st.markdown("##### Enlaces")
        st.markdown("\n".join(f"- [{k}]({v})" for k, v in ENLACES.items()))
