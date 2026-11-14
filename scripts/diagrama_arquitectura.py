"""Genera el diagrama de arquitectura RAG (SVG) usado en la app y en el póster.

Uso:  python scripts/diagrama_arquitectura.py   → assets/arquitectura_rag.svg
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
AZUL, AZUL2, TEAL, NAR, VERDE, MOR, GRIS = "#0B3C5D", "#0B4F6C", "#1F7A8C", "#E07A1F", "#2E8B57", "#7B4FA0", "#5B6770"


def caja(x, y, w, h, titulo, sub, color, relleno="#FFFFFF", texto=None, paso=None):
    texto = texto or color
    lineas = sub.split("|")
    s = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{relleno}" stroke="{color}" stroke-width="2.2"/>']
    ty = y + h / 2 - (len(lineas) * 15) / 2 - 4
    s.append(f'<text x="{x + w / 2}" y="{ty}" text-anchor="middle" font-weight="700" font-size="16.5" fill="{texto}">{titulo}</text>')
    for i, l in enumerate(lineas):
        s.append(f'<text x="{x + w / 2}" y="{ty + 21 + i * 16}" text-anchor="middle" font-size="12.8" '
                 f'fill="{"#E8F1F5" if relleno != "#FFFFFF" else "#3B4750"}">{l}</text>')
    if paso:
        s.append(f'<circle cx="{x + 14}" cy="{y + 14}" r="11" fill="{color}"/>'
                 f'<text x="{x + 14}" y="{y + 18.5}" text-anchor="middle" font-size="12" font-weight="700" fill="#fff">{paso}</text>')
    return "".join(s)


def flecha(x1, y1, x2, y2, color=GRIS, guion=False, etiqueta=None, ex=0, ey=-8):
    d = ' stroke-dasharray="6 5"' if guion else ""
    s = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2.2"{d} marker-end="url(#m{color[1:]})"/>'
    if etiqueta:
        s += f'<text x="{(x1 + x2) / 2 + ex}" y="{(y1 + y2) / 2 + ey}" text-anchor="middle" font-size="12" fill="{color}" font-style="italic">{etiqueta}</text>'
    return s


def ruta(d, color=GRIS, guion=False):
    g = ' stroke-dasharray="6 5"' if guion else ""
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.2"{g} marker-end="url(#m{color[1:]})"/>'


def svg():
    W, H = 1600, 640
    marcadores = "".join(
        f'<marker id="m{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for c in {GRIS, NAR, VERDE, MOR, TEAL, AZUL2})
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter, Arial, sans-serif">',
         f"<defs>{marcadores}</defs>", f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>']

    # Carril superior: consulta en línea
    p.append(f'<rect x="8" y="8" width="{W - 16}" height="300" rx="16" fill="#F2F7FA" stroke="#D3E3EC"/>')
    p.append(f'<text x="28" y="38" font-size="15" font-weight="700" fill="{AZUL}" letter-spacing="1">'
             f'FLUJO DE CONSULTA EN LÍNEA (ACA 2)</text>')
    xs, y, w, h = [24, 220, 416, 612, 808, 1004, 1200, 1396], 92, 176, 104
    nodos = [("Usuario", "Streamlit · chat|con historial", AZUL, AZUL, "#fff", "4"),
             ("Reescritura", "Gemini · T = 0|solo si hay historial", TEAL, "#FFFFFF", None, "3"),
             ("Embedding", "gemini-embedding-2|768 d · query", TEAL, "#FFFFFF", None, "1"),
             ("Qdrant Cloud", "Top-k · coseno|filtros de payload", AZUL2, AZUL2, "#fff", "1"),
             ("Umbral 0,645", "relevancia mínima|calibrada en ACA 1", MOR, "#FFFFFF", None, "3"),
             ("Prompt aumentado", "reglas + contexto [n]|+ evidencia dataset", TEAL, "#FFFFFF", None, "3"),
             ("Gemini 2.5 Flash", "T 0,2 · top-p 0,9|1.024 tokens · stream", NAR, NAR, "#fff", "2"),
             ("Respuesta", "citas verificadas|fuentes · latencias", VERDE, "#FFFFFF", None, "5")]
    for x, (t, s, c, r, tx, paso) in zip(xs, nodos):
        p.append(caja(x, y, w, h, t, s, c, r, tx))
    for a, b in zip(xs, xs[1:]):
        p.append(flecha(a + w + 2, y + h / 2, b - 4, y + h / 2))
    # Retorno a la interfaz y abstención
    p.append(ruta(f"M{xs[7] + w / 2},{y + h + 2} L{xs[7] + w / 2},{y + h + 52} L{xs[0] + w / 2},{y + h + 52} L{xs[0] + w / 2},{y + h + 6}", VERDE))
    p.append(f'<text x="{(xs[0] + xs[7] + w) / 2 + 330}" y="{y + h + 72}" text-anchor="middle" font-size="12.5" fill="{VERDE}" '
             f'font-style="italic">respuesta en streaming + panel de fuentes (documento, página, similitud)</text>')
    p.append(ruta(f"M{xs[4] + w / 2},{y - 2} L{xs[4] + w / 2},{y - 30} L{xs[7] + w / 2},{y - 30} L{xs[7] + w / 2},{y - 6}", MOR, True))
    p.append(f'<text x="{(xs[4] + xs[7] + w) / 2 + 20}" y="{y - 37}" text-anchor="middle" font-size="12.5" fill="{MOR}" '
             f'font-style="italic">sin evidencia → abstención sin llamar al LLM</text>')

    # Carril inferior: base de conocimiento y módulos
    y2 = 330
    p.append(f'<rect x="8" y="{y2}" width="1000" height="300" rx="16" fill="#F7F5FB" stroke="#E2DCEF"/>')
    p.append(f'<text x="28" y="{y2 + 30}" font-size="15" font-weight="700" fill="{MOR}" letter-spacing="1">'
             f'BASE DE CONOCIMIENTO (ACA 1, FUERA DE LÍNEA)</text>')
    p.append(caja(28, y2 + 56, 220, 96, "Corpus PDF", "Reglamento CUN · Guía MEN|Bienestar MEN · 327 págs.", MOR))
    p.append(caja(28, y2 + 176, 220, 96, "Dataset UCI", "4.424 estudiantes|SQL → 10 fichas", MOR))
    p.append(caja(290, y2 + 116, 210, 96, "Limpieza + chunking", "Recursive splitter|1.500 car. · 15 % solape", MOR))
    p.append(caja(542, y2 + 116, 210, 96, "Embeddings", "Gemini 768 d (principal)|E5 384 d (línea base)", MOR))
    p.append(caja(794, y2 + 116, 190, 96, "Batch upsert", "610 fragmentos|5,3 MB · UUID v5", MOR))
    p.append(flecha(250, y2 + 104, 288, y2 + 150, MOR))
    p.append(flecha(250, y2 + 224, 288, y2 + 180, MOR))
    p.append(flecha(502, y2 + 164, 540, y2 + 164, MOR))
    p.append(flecha(754, y2 + 164, 792, y2 + 164, MOR))
    p.append(ruta(f"M889,{y2 + 114} L889,{y2 - 4} L{xs[3] + w / 2},{y2 - 4} L{xs[3] + w / 2},{y + h + 6}", MOR, True))
    p.append(f'<text x="{xs[3] + w / 2 - 10}" y="{y2 - 10}" text-anchor="end" font-size="12.5" fill="{MOR}" font-style="italic">indexación del ACA 1 (colección reutilizada, sin reindexar)</text>')

    p.append(f'<rect x="1024" y="{y2}" width="568" height="300" rx="16" fill="#F3F9F5" stroke="#D5EBDD"/>')
    p.append(f'<text x="1044" y="{y2 + 30}" font-size="15" font-weight="700" fill="{VERDE}" letter-spacing="1">'
             f'MÓDULOS DE LA INTERFAZ (STREAMLIT CLOUD)</text>')
    p.append(caja(1044, y2 + 56, 254, 104, "Gestión documental", "inventario · explorador de chunks|búsqueda de prueba · carga|protegida con clave", VERDE))
    p.append(caja(1318, y2 + 56, 254, 104, "Alerta temprana", "regresión logística · 12 var.|probabilidad de deserción|→ consulta al RAG", VERDE))
    p.append(caja(1044, y2 + 182, 254, 96, "Evaluación", "Hit@k · MRR · LLM juez|robustez · latencia", VERDE))
    p.append(caja(1318, y2 + 182, 254, 96, "Configuración", "modelo · temperatura · top-p|Top-k · umbral · filtros", VERDE))
    p.append("</svg>")
    return "".join(p)


if __name__ == "__main__":
    (RAIZ / "assets").mkdir(exist_ok=True)
    (RAIZ / "assets" / "arquitectura_rag.svg").write_text(svg(), encoding="utf-8")
    print("assets/arquitectura_rag.svg")
