"""Conexión con el modelo generativo (Paso 2 del ACA 2): Google Gemini vía API.

Hiperparámetros por defecto y su justificación:
* temperature = 0.2  → respuestas estables y apegadas al contexto; un RAG normativo no necesita creatividad.
* top_p = 0.9        → recorta la cola de tokens poco probables sin volver la salida repetitiva.
* max_output_tokens = 1024 → suficiente para ~220 palabras con citas, con margen.
* razonamiento interno ("thinking") al mínimo → baja la latencia; la tarea es extractiva.
"""
from __future__ import annotations

import re
import time

from .embeddings import ERRORES_TRANSITORIOS


class ErrorLLM(RuntimeError):
    pass


class GeminiLLM:
    def __init__(self, api_key: str, modelo: str = "gemini-3.5-flash", respaldo=(), max_reintentos: int = 3):
        from google import genai

        if not api_key:
            raise ValueError("Falta GOOGLE_API_KEY para usar el LLM de Gemini.")
        self.cliente = genai.Client(api_key=api_key)
        self.modelo = modelo
        self.respaldo = [m for m in respaldo if m != modelo]
        self.max_reintentos = max_reintentos
        self.ultimo_uso: dict = {}
        self.modelo_usado = modelo

    def modelos_disponibles(self) -> list[str]:
        """Modelos de texto habilitados para la llave (se filtran los de audio, imagen o TTS)."""
        nombres = []
        try:
            for m in self.cliente.models.list():
                n = m.name.removeprefix("models/")
                acciones = getattr(m, "supported_actions", None) or []
                if "generateContent" in acciones and n.startswith("gemini") and not re.search(
                        r"tts|image|audio|live|embedding|robotics|computer", n):
                    nombres.append(n)
        except Exception:
            pass
        return sorted(set(nombres), reverse=True)

    @staticmethod
    def _config(modelo, system, temperatura, top_p, max_tokens, con_thinking=True):
        from google.genai import types

        kw = dict(system_instruction=system, temperature=temperatura, top_p=top_p, max_output_tokens=max_tokens)
        if con_thinking:
            if modelo.startswith("gemini-2.5-flash"):
                kw["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            elif modelo.startswith("gemini-2.5-pro"):
                kw["thinking_config"] = types.ThinkingConfig(thinking_budget=128)
            elif modelo.startswith("gemini-3") or "latest" in modelo:
                kw["thinking_config"] = types.ThinkingConfig(thinking_level="minimal")
        return types.GenerateContentConfig(**kw)

    @staticmethod
    def _contenidos(mensajes):
        from google.genai import types

        return [types.Content(role="model" if m["role"] == "assistant" else "user",
                              parts=[types.Part.from_text(text=m["content"])]) for m in mensajes]

    def _uso(self, r):
        u = getattr(r, "usage_metadata", None)
        if u:
            self.ultimo_uso = {"tokens_entrada": u.prompt_token_count or 0,
                               "tokens_salida": u.candidates_token_count or 0}

    def _intentar(self, fn):
        """Ejecuta fn(modelo, con_thinking) con reintentos, sin thinking si el modelo no lo admite
        y con modelos de respaldo si el principal no existe o agotó su cuota."""
        ultimo = None
        for modelo in [self.modelo, *self.respaldo]:
            con_thinking, intento = True, 0
            while True:
                try:
                    self.modelo_usado = modelo
                    salida = fn(modelo, con_thinking)
                    if modelo != self.modelo:          # recuerda el modelo que sí respondió
                        self.respaldo = [m for m in [self.modelo, *self.respaldo] if m != modelo]
                        self.modelo = modelo
                    return salida
                except Exception as e:
                    ultimo, msg = e, str(e)
                    if "thinking" in msg.lower() and con_thinking:
                        con_thinking = False      # el modelo no admite esa configuración: reintenta sin ella
                        continue
                    if "404" in msg or "NOT_FOUND" in msg:
                        break                     # el modelo no existe: pasa al de respaldo
                    if any(x in msg for x in ERRORES_TRANSITORIOS) and intento < self.max_reintentos:
                        intento += 1
                        time.sleep(min(20, 2 ** intento))
                        continue
                    if any(x in msg for x in ERRORES_TRANSITORIOS):
                        break                     # cuota agotada o servicio caído: prueba otro modelo
                    raise ErrorLLM(msg) from e
        raise ErrorLLM(str(ultimo))

    def generar(self, mensajes, system=None, temperatura=0.2, top_p=0.9, max_tokens=1024) -> str:
        def fn(modelo, th):
            r = self.cliente.models.generate_content(
                model=modelo, contents=self._contenidos(mensajes),
                config=self._config(modelo, system, temperatura, top_p, max_tokens, th))
            self._uso(r)
            return (r.text or "").strip()
        return self._intentar(fn)

    def generar_stream(self, mensajes, system=None, temperatura=0.2, top_p=0.9, max_tokens=1024):
        """Generador de fragmentos de texto. El primer fragmento se pide dentro de los reintentos,
        así los errores de modelo o cuota se resuelven antes de mostrar nada al usuario."""
        def fn(modelo, th):
            it = iter(self.cliente.models.generate_content_stream(
                model=modelo, contents=self._contenidos(mensajes),
                config=self._config(modelo, system, temperatura, top_p, max_tokens, th)))
            primero = next(it)
            return primero, it

        primero, it = self._intentar(fn)

        def flujo():
            ultimo = primero
            if primero.text:
                yield primero.text
            for trozo in it:
                ultimo = trozo
                if trozo.text:
                    yield trozo.text
            self._uso(ultimo)
        return flujo()


class OfflineLLM:
    """Respuesta extractiva sin red: SOLO para pruebas automáticas del flujo y de la interfaz."""

    modelo = modelo_usado = "offline-extractivo"

    def __init__(self):
        self.ultimo_uso = {}

    def modelos_disponibles(self):
        return [self.modelo]

    def generar(self, mensajes, system=None, **_):
        texto = mensajes[-1]["content"]
        if texto.startswith("Reescribe"):
            return texto.rsplit("ÚLTIMA PREGUNTA:", 1)[-1].strip()
        bloques = re.findall(r"\[(\d+)\] [^\n]*\n(.+?)(?=\n\n\[\d+\]|\n\nPREGUNTA DEL USUARIO)", texto, re.S)
        partes = []
        for n, cuerpo in bloques[:2]:
            frase = re.split(r"(?<=[.;:])\s", " ".join(cuerpo.split()))[0][:260]
            partes.append(f"- {frase} [{n}]")
        return "[Modo de prueba sin LLM] Fragmentos más relevantes:\n" + "\n".join(partes)

    def generar_stream(self, mensajes, system=None, **kw):
        for palabra in self.generar(mensajes, system).split(" "):
            yield palabra + " "


def crear_llm(config, modelo: str | None = None):
    if config.proveedor_llm == "offline":
        return OfflineLLM()
    return GeminiLLM(config.google_api_key, modelo or config.llm_model, config.llm_respaldo)
