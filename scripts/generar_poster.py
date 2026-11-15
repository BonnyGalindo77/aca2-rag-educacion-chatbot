"""Genera el póster infografía (A3, PDF con hipervínculos) a partir de los resultados del proyecto.

Lee:  resultados/metricas_modelo.json · resultados/evaluacion_aca2.json (si existe) · assets/*.png · enlaces.json
Uso:  python scripts/generar_poster.py   → entregables/Poster_ACA2_RAG_Educacion.pdf
Requiere: playwright + Chrome/Chromium, qrcode.
"""
from __future__ import annotations

import asyncio
import base64
import html
import io
import json
import sys
from pathlib import Path

import qrcode
import re
import qrcode.image.svg

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "entregables"


def cargar(ruta):
    p = RAIZ / ruta
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def img64(ruta):
    p = RAIZ / ruta
    if not p.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def qr_svg(url):
    img = qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage, box_size=10, border=1)
    b = io.BytesIO()
    img.save(b)
    s = b.getvalue().decode()
    return s[s.index("<svg"):].replace("<svg ", '<svg class="qr" ', 1)


def n(x, d=2, pct=False):
    if x is None or x == "":
        return "—"
    if pct:
        return f"{x * 100:.0f} %".replace(".", ",")
    return f"{x:.{d}f}".replace(".", ",")


def construir_html():
    mm = cargar("resultados/metricas_modelo.json")
    ev = cargar("resultados/evaluacion_aca2.json")
    enl = cargar("enlaces.json")
    rec, gen = ev.get("recuperacion", {}), ev.get("generacion", {})
    p = mm.get("prueba_20pct", {})
    url_app, url_video, url_repo = enl.get("app", ""), enl.get("video", ""), enl.get("repo", "")
    logo = img64("assets/logo_cun.png")
    arq = (RAIZ / "assets" / "arquitectura_rag.svg").read_text(encoding="utf-8")
    cap_chat, cap_gest = img64("assets/captura_chat.png"), img64("assets/captura_gestion.png")
    fig_eval = img64("resultados/fig_evaluacion.png")
    lat_llm = gen.get("latencia_llm_ms_mediana")
    modelo_llm = "gemini-3.5-flash"
    modelo_eval = ev.get("modelo_llm") or modelo_llm

    ejemplo = next((c for c in ev.get("casos", []) if c.get("id") == "G1" and c.get("estado") == "ok"), None)
    if ejemplo:
        resp = html.escape(ejemplo["respuesta"])
        resp = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", resp)
        resp = re.sub(r"(^|\n)\s*[\*\-]\s+", r"\1• ", resp).replace("\n", "<br>")
        if len(resp) > 430:
            resp = resp[:430].rsplit(" ", 1)[0] + "…"
        fuentes = " · ".join(html.escape(f) for f in ejemplo["fuentes"][:3])
        ejemplo_html = (f'<div class="q">“{html.escape(ejemplo["pregunta"])}”</div><div class="a">{resp}</div>'
                        f'<div class="src">Fuentes: {fuentes}</div>')
    else:
        ejemplo_html = '<div class="pend">Se completa con una respuesta real del chatbot tras ejecutar el notebook.</div>'

    def enlace(nombre, url):
        if not url:
            return ""  # sin enlace: el bloque no se muestra
        return (f'<div class="link">{qr_svg(url)}<div><b>{nombre}</b><br><a href="{url}">{html.escape(url)}</a></div></div>')

    cv = "".join(f"<tr><td>{c['modelo']}</td><td>{n(c['roc_auc'], 3)}</td><td>{n(c['f1'], 3)}</td><td>{n(c['recall'], 3)}</td></tr>"
                 for c in mm.get("comparacion_cv5", []))
    logo_html = f'<img class="logo" src="{logo}">' if logo else '<div class="logo ph">Logo CUN</div>'

    return f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8"><title>Póster ACA 2 · RAG Educación</title>
<style>
@page {{ size: 297mm 420mm; margin: 0; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; width:297mm; height:420mm; font-family: Inter, Arial, sans-serif; color:#1F2A33; background:#EEF3F6;
       font-size:8.1pt; line-height:1.32; }}
a {{ color:#0B4F6C; text-decoration:underline; word-break:break-all; }}
.head {{ background: linear-gradient(115deg,#082E47 0%,#0B4F6C 62%,#1F7A8C 100%); color:#fff; padding:4.5mm 9mm 4mm; display:flex;
        gap:7mm; align-items:center; }}
.logo {{ height:19mm; background:#fff; border-radius:3mm; padding:2.2mm 3mm; }}
.logo.ph {{ width:42mm; display:flex; align-items:center; justify-content:center; color:#0B4F6C; font-weight:700; }}
.kicker {{ font-size:7.4pt; letter-spacing:.14em; text-transform:uppercase; color:#BFDCE6; }}
h1 {{ margin:1mm 0 0.8mm; font-size:19.5pt; line-height:1.12; font-weight:800; letter-spacing:-.01em; }}
h1 span {{ color:#FFB347; }}
.team {{ font-size:9pt; color:#E6F1F5; }} .team b {{ color:#fff; }}
.grid {{ display:grid; grid-template-columns: repeat(12, 1fr); gap:3mm; padding:3.4mm 6mm 0; }}
.card {{ background:#fff; border-radius:2.6mm; padding:2.8mm 3.6mm; border-top:1.3mm solid #0B4F6C; box-shadow:0 .4mm 1.2mm rgba(0,0,0,.06); }}
.card h2 {{ margin:0 0 1.8mm; font-size:9.4pt; letter-spacing:.07em; text-transform:uppercase; color:#0B3C5D; }}
.c-or {{ border-top-color:#E07A1F; }} .c-mo {{ border-top-color:#7B4FA0; }} .c-ve {{ border-top-color:#2E8B57; }} .c-te {{ border-top-color:#1F7A8C; }}
.s3 {{ grid-column: span 3; }} .s4 {{ grid-column: span 4; }} .s5 {{ grid-column: span 5; }} .s6 {{ grid-column: span 6; }}
.s7 {{ grid-column: span 7; }} .s8 {{ grid-column: span 8; }} .s12 {{ grid-column: span 12; }}
p {{ margin:0 0 1.4mm; }}
.pregunta {{ background:#FFF4E6; border-left:1mm solid #E07A1F; padding:1.8mm 2.4mm; font-style:italic; color:#5A3510; border-radius:1mm; }}
.kpis {{ display:grid; grid-template-columns:1fr 1fr; gap:2mm; }}
.kpi {{ background:#F2F7FA; border-radius:1.6mm; padding:1.8mm 2mm; }}
.kpi .v {{ font-size:15pt; font-weight:800; color:#E07A1F; line-height:1.05; }}
.kpi .l {{ font-size:7.2pt; color:#3B4750; }}
table {{ width:100%; border-collapse:collapse; font-size:7.8pt; }}
th {{ text-align:left; font-size:7pt; text-transform:uppercase; letter-spacing:.05em; color:#5B6770; border-bottom:.3mm solid #C9D6DE; padding:.8mm 1mm; }}
td {{ padding:.9mm 1mm; border-bottom:.2mm solid #E4EBEF; vertical-align:top; }}
td b {{ color:#0B3C5D; }}
.tag {{ display:inline-block; background:#E8F1F5; color:#0B3C5D; border-radius:1mm; padding:0 1.2mm; font-size:6.8pt; }}
.pasos {{ display:grid; grid-template-columns:repeat(5,1fr); gap:1.6mm; }}
.paso {{ background:#F2F7FA; border-radius:1.6mm; padding:1.8mm; position:relative; }}
.paso .nn {{ display:inline-block; background:#0B4F6C; color:#fff; border-radius:50%; width:5.2mm; height:5.2mm; text-align:center;
            line-height:5.2mm; font-weight:700; font-size:7.6pt; margin-right:1mm; }}
.paso b {{ font-size:8.2pt; color:#0B3C5D; }} .paso div {{ font-size:7.2pt; margin-top:.8mm; color:#3B4750; }}
.arq svg {{ width:100%; height:auto; display:block; }}
.shots {{ display:grid; grid-template-columns:1fr 1fr; gap:2mm; }}
.shots img {{ width:100%; height:37mm; object-fit:cover; object-position:top left; border-radius:1.4mm; border:.3mm solid #D3E3EC; display:block; }}
.shots .cap {{ font-size:6.8pt; color:#5B6770; margin-top:.6mm; }}
.shotph {{ height:40mm; border:.4mm dashed #9DB5C2; border-radius:1.4mm; display:flex; align-items:center; justify-content:center;
          color:#5B6770; font-size:7.6pt; text-align:center; padding:2mm; }}
.q {{ font-weight:700; color:#0B3C5D; margin-bottom:1mm; }}
.a {{ background:#F4FAF6; border-left:1mm solid #2E8B57; padding:1.6mm 2.2mm; border-radius:1mm; font-size:7.6pt; }}
.src {{ font-size:6.8pt; color:#5B6770; margin-top:1mm; }}
.pend {{ color:#9A6A2F; font-style:italic; }}
ul {{ margin:0; padding-left:4mm; }} li {{ margin-bottom:1.1mm; }}
.links {{ display:flex; flex-direction:column; gap:1.6mm; }}
.link {{ display:flex; gap:2.6mm; align-items:center; font-size:7.6pt; }}
.qr {{ width:16mm; height:16mm; flex:none; }}
.qrph {{ width:19mm; height:19mm; border:.4mm dashed #9DB5C2; display:flex; align-items:center; justify-content:center; color:#9DB5C2; flex:none; }}
.refs p {{ font-size:6.3pt; line-height:1.3; margin:0 0 1mm; padding-left:3mm; text-indent:-3mm; }}
.foot {{ position:absolute; bottom:0; left:0; right:0; background:#082E47; color:#BFDCE6; font-size:6.6pt; padding:1.4mm 9mm;
        display:flex; justify-content:space-between; }}
.met {{ display:grid; grid-template-columns:repeat(4,1fr); gap:1.6mm; margin:1.4mm 0 2mm; }}
.met .kpi .v {{ font-size:11.5pt; color:#0B4F6C; }}
.met .kpi.o .v {{ color:#E07A1F; }} .met .kpi.g .v {{ color:#2E8B57; }}
.nota {{ font-size:6.8pt; color:#5B6770; }}
</style></head><body>
<div class="head">{logo_html}
<div><div class="kicker">Corporación Unificada Nacional de Educación Superior · Especialización en Inteligencia Artificial · Computación en la Nube · ACA 2</div>
<h1>Chatbot RAG de permanencia estudiantil:<br><span>Gemini API + Qdrant Cloud + Streamlit</span></h1>
<div class="team"><b>Bonny Darhyll Brayan Galindo Santos · Camilo José Rodríguez Ayubi · Andrés Alberto López Herrera</b><br>
Docente: Hamilton Rivera Flor · Pilar 2: Educación · Noviembre de 2026</div></div></div>

<div class="grid">
 <div class="card s4"><h2>Objetivo general</h2>
  <p>Integrar un modelo de lenguaje en la nube (Google Gemini vía API) al sistema de recuperación vectorial del ACA 1 en Qdrant Cloud y
  desplegar una interfaz en Streamlit para conversar con un chatbot RAG, gestionar los documentos base y supervisar el índice.</p></div>
 <div class="card s5 c-or"><h2>Situación problema</h2>
  <p>Cerca de la mitad de quienes ingresan a la universidad en Colombia no terminan (MEN, 2015). Las reglas y apoyos que podrían
  retenerlos están dispersos en reglamentos y guías de cientos de páginas.</p>
  <div class="pregunta">¿Cómo articular un LLM con una base vectorial serverless para construir un chatbot RAG confiable, con una interfaz que
  facilite la administración documental, la consulta en lenguaje natural y la reducción de alucinaciones?</div></div>
 <div class="card s3 c-te"><h2>En cifras</h2><div class="kpis">
  <div class="kpi"><div class="v">610</div><div class="l">fragmentos en Qdrant Cloud</div></div>
  <div class="kpi"><div class="v">32,1 %</div><div class="l">deserción en la cohorte UCI</div></div>
  <div class="kpi"><div class="v">{n(rec.get('MRR'), 2)}</div><div class="l">MRR de recuperación</div></div>
  <div class="kpi"><div class="v">{n(p.get('roc_auc'), 2)}</div><div class="l">ROC-AUC alerta temprana</div></div></div></div>

 <div class="card s5 c-mo"><h2>Datos</h2>
  <table><tr><th>Fuente</th><th>Aporta</th><th>Chunks</th></tr>
  <tr><td><b>Reglamento Estudiantil CUN</b><br>Acuerdo 002 de 2011</td><td>Matrícula, retiros, reintegros, cancelaciones, evaluación, grados</td><td>113</td></tr>
  <tr><td><b>Guía de permanencia</b><br>MEN, 2015</td><td>Modelo de gestión y caja de herramientas</td><td>383</td></tr>
  <tr><td><b>Lineamientos de bienestar</b><br>MEN, 2016</td><td>Dimensiones y componentes del bienestar</td><td>104</td></tr>
  <tr><td><b>Dataset UCI</b> (Realinho et al., 2021)</td><td>4.424 estudiantes · 36 variables → SQL → fichas y modelo de riesgo</td><td>10</td></tr></table></div>
 <div class="card s7"><h2>Metodología (ruta del ACA 2)</h2><div class="pasos">
  <div class="paso"><span class="nn">1</span><b>Conexión vectorial</b><div>Colección del ACA 1 reutilizada; consulta con gemini-embedding-2 (768 d) y Top-k con filtros.</div></div>
  <div class="paso"><span class="nn">2</span><b>Configuración LLM</b><div>{html.escape(modelo_llm)} · T 0,2 · top-p 0,9 · 2.048 tokens · llaves en st.secrets.</div></div>
  <div class="paso"><span class="nn">3</span><b>Pipeline RAG</b><div>Umbral 0,645, contexto numerado, citas [n], memoria y reescritura de seguimiento.</div></div>
  <div class="paso"><span class="nn">4</span><b>Interfaz</b><div>Streamlit Cloud: chat con historial, fuentes, gestión documental y alerta temprana.</div></div>
  <div class="paso"><span class="nn">5</span><b>Evaluación</b><div>Hit@k, MRR, LLM como juez, abstención, inyección y latencia por etapa.</div></div>
 </div></div>

 <div class="card s8 c-te arq"><h2>Arquitectura RAG end-to-end</h2>{arq}</div>

 <div class="card s4"><h2>Modelos</h2>
  <table><tr><th>Componente</th><th>Elección y justificación</th></tr>
  <tr><td><b>Embeddings</b></td><td>gemini-embedding-2 · 768 d · mismo formato del ACA 1 para que consulta e índice sean comparables</td></tr>
  <tr><td><b>LLM (nube)</b></td><td>{html.escape(modelo_llm)}: capa gratuita, buen español, despliegue sin GPU; respaldo automático ante cuota 429</td></tr>
  <tr><td><b>Alerta temprana</b></td><td>Regresión logística elegida por validación cruzada (5 pliegues) frente a Random Forest y Gradient Boosting</td></tr></table>
  <table style="margin-top:1.6mm"><tr><th>Modelo (CV-5)</th><th>ROC-AUC</th><th>F1</th><th>Recall</th></tr>{cv}</table>
  <div class="nota">Prueba independiente (20 %): ROC-AUC {n(p.get('roc_auc'), 3)} · recall {n(p.get('recall'), 3)} · F1 {n(p.get('f1'), 3)}.
  Variable más influyente: asignaturas aprobadas en el primer semestre.</div></div>
 <div class="card s5 c-or"><h2>Resultados</h2>
  <div class="met">
   <div class="kpi"><div class="v">{n(rec.get('Hit@1'), 2)}</div><div class="l">Hit@1 (14 consultas)</div></div>
   <div class="kpi"><div class="v">{n(rec.get('Hit@3'), 2)}</div><div class="l">Hit@3</div></div>
   <div class="kpi o"><div class="v">{n(gen.get('juez_fidelidad_media'), 2)}</div><div class="l">fidelidad (juez 1–5)</div></div>
   <div class="kpi o"><div class="v">{n(gen.get('juez_relevancia_media'), 2)}</div><div class="l">relevancia (juez 1–5)</div></div>
   <div class="kpi g"><div class="v">{n(gen.get('comportamiento_correcto'), pct=True)}</div><div class="l">comportamiento correcto (10 casos)</div></div>
   <div class="kpi g"><div class="v">{n(gen.get('abstencion_correcta_fuera_dominio'), pct=True)}</div><div class="l">abstención sin evidencia</div></div>
   <div class="kpi"><div class="v">{n(rec.get('lat_qdrant_ms_mediana'), 0)} ms</div><div class="l">búsqueda Qdrant (mediana)</div></div>
   <div class="kpi"><div class="v">{n((lat_llm or 0) / 1000, 1) if lat_llm else '—'} s</div><div class="l">generación LLM (mediana)</div></div>
  </div>
  <div class="nota" style="margin:-0.6mm 0 1.6mm">Corrida en Colab del {html.escape(ev.get("fecha", "")[:10])}: 14 consultas de recuperación y
  10 casos de generación y robustez, respondidos con {html.escape(modelo_eval)} (respaldo automático del cliente); juez: Gemini, temperatura 0.</div>
  {f'<img src="{fig_eval}" style="width:100%;margin-top:1.5mm;border-radius:1mm">' if fig_eval else ''}
</div>

 <div class="card s7 c-ve"><h2>Vista previa de la aplicación (Streamlit Community Cloud)</h2><div class="shots">
  {f'<div><img src="{cap_chat}"><div class="cap">Chatbot: respuesta en streaming con citas, fuentes, página, similitud y latencias.</div></div>' if cap_chat else '<div class="shotph">Captura del chat (se agrega tras el despliegue)</div>'}
  {f'<div><img src="{cap_gest}"><div class="cap">Gestión documental: estado del clúster, inventario, explorador de fragmentos y búsqueda de prueba.</div></div>' if cap_gest else '<div class="shotph">Captura de gestión documental (se agrega tras el despliegue)</div>'}
 </div>
 <div class="q" style="font-size:7.6pt;margin-top:2mm">Ejemplo real de interacción con cita</div>{ejemplo_html}</div>

 <div class="card s5"><h2>Conclusiones</h2><ul>
  <li><b>Ciclo RAG completo a costo cero:</b> la colección del ACA 1 se reutilizó sin reindexar; solo se necesitó vectorizar la consulta con el mismo modelo.</li>
  <li><b>Confiabilidad por capas:</b> umbral de relevancia, prompt con reglas de citación y verificación posterior de citas reducen las alucinaciones; sin evidencia, el asistente se abstiene.</li>
  <li><b>La latencia la domina el LLM:</b> buscar en Qdrant toma milisegundos; por eso la respuesta se muestra en streaming.</li>
  <li><b>Norma + datos + predicción:</b> el modelo de alerta temprana convierte el riesgo de un estudiante en una consulta al chatbot con las normas y apoyos aplicables.</li>
  <li><b>Límites:</b> cifras de una institución portuguesa, reglamento 2011 con modificaciones, evaluación con juez automático.</li></ul></div>
 <div class="card s3 c-or"><h2>Enlaces</h2><div class="links">{enlace("Aplicación desplegada", url_app)}{enlace("Video explicativo", url_video)}{enlace("Repositorio GitHub", url_repo)}</div></div>
 <div class="card s4 refs"><h2>Referencias (APA 7)</h2>
  <p>Google. (2026). <i>Gemini API documentation</i>. https://ai.google.dev/gemini-api/docs</p>
  <p>Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., … Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. <i>Advances in Neural Information Processing Systems, 33</i>, 9459–9474.</p>
  <p>Ministerio de Educación Nacional. (2015). <i>Guía para la implementación del modelo de gestión de permanencia y graduación estudiantil en IES</i>.</p>
  <p>Ministerio de Educación Nacional. (2016). <i>Lineamientos de política de bienestar para instituciones de educación superior</i>.</p>
  <p>Qdrant Solutions GmbH. (2024). <i>Qdrant documentation</i>. https://qdrant.tech/documentation/</p>
  <p>Realinho, V., Vieira Martins, M., Machado, J., &amp; Baptista, L. (2021). <i>Predict students' dropout and academic success</i> [Conjunto de datos]. UCI. https://doi.org/10.24432/C5MC89</p>
  <p>Streamlit Inc. (2024). <i>Streamlit documentation</i>. https://docs.streamlit.io/</p>
  <p>Zheng, L., Chiang, W.-L., Sheng, Y., … Stoica, I. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. <i>NeurIPS, 36</i>.</p></div>
</div>
<div class="foot"><span>ACA 2 · Computación en la Nube · CUN 2026</span><span>Qdrant Cloud · Gemini API · Streamlit Community Cloud · capa gratuita</span></div>
</body></html>"""


async def a_pdf(html_txt, destino):
    from playwright.async_api import async_playwright

    chrome = next((c for c in ("/usr/bin/google-chrome", "/usr/bin/chromium") if Path(c).exists()), None)
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path=chrome, args=["--no-sandbox"]) if chrome else await p.chromium.launch()
        pg = await b.new_page()
        await pg.set_content(html_txt, wait_until="networkidle")
        await pg.wait_for_timeout(600)
        alto = await pg.evaluate("Math.max(...[...document.querySelectorAll('.card')].map(e => e.getBoundingClientRect().bottom)) + document.querySelector('.foot').getBoundingClientRect().height")
        await pg.pdf(path=str(destino), width="297mm", height="420mm", print_background=True)
        await b.close()
        return alto


if __name__ == "__main__":
    SALIDA.mkdir(exist_ok=True)
    h = construir_html()
    (SALIDA / "poster.html").write_text(h, encoding="utf-8")
    alto = asyncio.run(a_pdf(h, SALIDA / "Poster_ACA2_RAG_Educacion.pdf"))
    limite = 420 / 25.4 * 96
    print(f"PDF generado · alto del contenido {alto}px (límite A3 ≈ {limite:.0f}px)", "⚠ desborda" if alto > limite else "")
    sys.exit(0)
