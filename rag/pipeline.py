"""Canalización RAG completa (Pasos 1 a 3 del ACA 2).

pregunta ─► (reescritura con historial) ─► embedding de consulta (Gemini) ─► Top-k en Qdrant Cloud
        ─► umbral de relevancia ─► contexto numerado con citas ─► LLM Gemini ─► verificación de citas
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .prompts import (PLANTILLA_PREGUNTA, PLANTILLA_REESCRITURA, RESPUESTA_SIN_EVIDENCIA, SYSTEM_PROMPT,
                      construir_contexto, formatear_cita)


@dataclass
class Resultado:
    pregunta: str
    consulta: str
    estado: str = "ok"                 # ok | sin_evidencia | error_llm
    respuesta: str = ""
    flujo: object = None               # generador de texto cuando se pide streaming
    fuentes: list = field(default_factory=list)
    descartadas: list = field(default_factory=list)
    tiempos: dict = field(default_factory=dict)
    modelo: str = ""
    uso: dict = field(default_factory=dict)
    citas: dict = field(default_factory=dict)
    error: str = ""


def analizar_citas(texto: str, n_fuentes: int) -> dict:
    numeros = {int(n) for grupo in re.findall(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]", texto)
               for n in re.findall(r"\d+", grupo)}
    validas = {n for n in numeros if 1 <= n <= n_fuentes}
    normal = re.sub(r"([.!?])\s*((?:\[\d[^\]]*\]\s*)+)", r" \2\1", texto)   # "frase. [1]" → "frase [1]."
    oraciones = [o for o in re.split(r"(?<=[.!?])\s+|\n+", normal) if len(o.strip()) > 25]
    con_cita = [o for o in oraciones if re.search(r"\[\d+", o)]
    return {"citadas": sorted(validas), "invalidas": sorted(numeros - validas),
            "cobertura": round(len(con_cita) / len(oraciones), 3) if oraciones else 0.0,
            "abstencion": bool(re.search(r"no encontr[ée]", texto, re.I))}


def _fuente_dict(i, p, umbral):
    c = p.payload
    return {"n": i, "score": round(float(p.score), 4), "supera_umbral": p.score >= umbral,
            "fuente_id": c.get("fuente_id"), "titulo": c.get("titulo"), "entidad": c.get("entidad"),
            "anio": c.get("anio"), "pagina": c.get("pagina"), "tipo_documento": c.get("tipo_documento"),
            "factores_riesgo": c.get("factores_riesgo", []), "evidencia_dataset": c.get("evidencia_dataset", []),
            "url": c.get("url", ""), "texto": c.get("texto", ""), "cita": formatear_cita(c)}


class RAGPipeline:
    def __init__(self, store, embedder, llm, config):
        self.store, self.embedder, self.llm, self.config = store, embedder, llm, config

    # --- Paso 1: recuperación ---------------------------------------------------------------
    def recuperar(self, consulta: str, k: int | None = None, umbral: float | None = None, filtros=None):
        k = k or self.config.top_k
        umbral = self.config.umbral if umbral is None else umbral
        t0 = time.perf_counter()
        vec = self.embedder.consulta(consulta)
        t1 = time.perf_counter()
        puntos = self.store.buscar(vec, k=k, filtros=filtros)
        t2 = time.perf_counter()
        utiles = [p for p in puntos if p.score >= umbral]
        return puntos, utiles, {"embedding_ms": (t1 - t0) * 1000, "qdrant_ms": (t2 - t1) * 1000}

    # --- memoria conversacional ---------------------------------------------------------------
    def _historial(self, historial):
        turnos = [m for m in (historial or []) if m.get("role") in ("user", "assistant") and m.get("content")]
        turnos = turnos[-2 * self.config.turnos_memoria:]
        while turnos and turnos[0]["role"] != "user":
            turnos.pop(0)
        return [{"role": m["role"], "content": m["content"]} for m in turnos]

    def reescribir(self, pregunta: str, historial) -> str:
        """Convierte preguntas de seguimiento ("¿y cuál es el plazo?") en consultas autónomas."""
        turnos = self._historial(historial)
        if not turnos:
            return pregunta
        texto = "\n".join(f"{'Usuario' if m['role'] == 'user' else 'Asistente'}: {m['content'][:600]}"
                          for m in turnos[-4:])
        try:
            r = self.llm.generar([{"role": "user", "content": PLANTILLA_REESCRITURA.format(
                historial=texto, pregunta=pregunta)}], temperatura=0.0, max_tokens=120)
            r = r.strip().strip('"').splitlines()[0] if r.strip() else ""
            return r if 8 < len(r) < 400 else pregunta
        except Exception:
            return pregunta

    # --- Pasos 2 y 3: generación aumentada ----------------------------------------------------
    def responder(self, pregunta: str, historial=None, k=None, umbral=None, filtros=None, reescribir=True,
                  stream=False, temperatura=None, top_p=None, max_tokens=None) -> Resultado:
        cfg = self.config
        umbral = cfg.umbral if umbral is None else umbral
        t_inicio = time.perf_counter()

        consulta = self.reescribir(pregunta, historial) if reescribir and historial else pregunta
        t_reesc = (time.perf_counter() - t_inicio) * 1000
        try:
            puntos, utiles, tiempos = self.recuperar(consulta, k=k, umbral=umbral, filtros=filtros)
        except Exception as e:   # credencial inválida, clúster suspendido o cuota agotada en la búsqueda
            msg = str(e)
            causa = ("la API key de Google no es válida" if "API_KEY_INVALID" in msg or "API key not valid" in msg
                     else "se agotó la cuota gratuita de embeddings" if "429" in msg or "RESOURCE_EXHAUSTED" in msg
                     else "no hubo respuesta del clúster de Qdrant Cloud")
            return Resultado(pregunta=pregunta, consulta=consulta, estado="error_busqueda", error=msg[:400],
                             respuesta=f"No pude consultar la base vectorial en este momento: {causa}. "
                                       "Intenta de nuevo en unos minutos.",
                             tiempos={"total_ms": (time.perf_counter() - t_inicio) * 1000})
        tiempos["reescritura_ms"] = t_reesc

        res = Resultado(pregunta=pregunta, consulta=consulta, tiempos=tiempos,
                        fuentes=[_fuente_dict(i, p, umbral) for i, p in enumerate(utiles, 1)],
                        descartadas=[_fuente_dict(0, p, umbral) for p in puntos if p.score < umbral])

        if not utiles:   # abstención: no se llama al LLM si no hay evidencia suficiente
            res.estado, res.respuesta = "sin_evidencia", RESPUESTA_SIN_EVIDENCIA
            res.tiempos["llm_ms"] = 0.0
            res.tiempos["total_ms"] = (time.perf_counter() - t_inicio) * 1000
            res.citas = analizar_citas(res.respuesta, 0)
            return res

        mensajes = self._historial(historial) + [{"role": "user", "content": PLANTILLA_PREGUNTA.format(
            contexto=construir_contexto(utiles), pregunta=pregunta if consulta == pregunta else
            f"{pregunta}\n(Interpretación autónoma de la pregunta: {consulta})")}]
        params = dict(system=SYSTEM_PROMPT, temperatura=cfg.temperatura if temperatura is None else temperatura,
                      top_p=cfg.top_p if top_p is None else top_p, max_tokens=max_tokens or cfg.max_tokens)

        t_llm = time.perf_counter()
        try:
            if stream:
                base = self.llm.generar_stream(mensajes, **params)
                res.tiempos["primer_token_ms"] = (time.perf_counter() - t_llm) * 1000

                def flujo():
                    partes = []
                    for trozo in base:
                        partes.append(trozo)
                        yield trozo
                    self._cerrar(res, "".join(partes), t_llm, t_inicio)
                res.flujo = flujo()
            else:
                self._cerrar(res, self.llm.generar(mensajes, **params), t_llm, t_inicio)
        except Exception as e:
            res.estado, res.error = "error_llm", str(e)[:400]
            res.respuesta = ("No pude contactar el modelo de lenguaje en este momento (posible límite de cuota de la "
                             "capa gratuita). Te muestro abajo los fragmentos recuperados para que los consultes "
                             "directamente.")
            res.tiempos["total_ms"] = (time.perf_counter() - t_inicio) * 1000
        return res

    def _cerrar(self, res: Resultado, texto: str, t_llm: float, t_inicio: float):
        res.respuesta = texto.strip()
        res.modelo = getattr(self.llm, "modelo_usado", "")
        res.uso = dict(getattr(self.llm, "ultimo_uso", {}) or {})
        res.tiempos["llm_ms"] = (time.perf_counter() - t_llm) * 1000
        res.tiempos["total_ms"] = (time.perf_counter() - t_inicio) * 1000
        res.citas = analizar_citas(res.respuesta, len(res.fuentes))
