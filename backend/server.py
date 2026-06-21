"""FastAPI server exposing the voice assistant pipeline over WebSocket.

Protocol (JSON messages from client -> server):
  - {"type": "audio", "data": "<base64 PCM float32 mono 16kHz>"}
  - {"type": "config", "config": {...}}            # update runtime config
  - {"type": "stop"}                                # stop current recording

Messages from server -> client:
  - {"type": "status", "state": "idle|listening|transcribing|thinking|speaking"}
  - {"type": "transcript", "text": "..."}
  - {"type": "response", "text": "..."}
  - {"type": "audio", "data": "<base64 PCM float32 mono>", "sample_rate": 22050}
  - {"type": "error", "message": "..."}
  - {"type": "ready"}
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import struct
import time
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from . import commands, llm, stt, tts, wakeword
from .config import load_config, resolve_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("voice-assistant.server")

app = FastAPI(title="Voice Assistant Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG = load_config()
SAMPLE_RATE = int(CONFIG["sample_rate"])
SILENCE_THRESHOLD = float(CONFIG["silence_threshold"])
SILENCE_DURATION_MS = int(CONFIG["silence_duration_ms"])
MAX_RECORDING_MS = int(CONFIG["max_recording_ms"])
MIN_RECORDING_MS = int(CONFIG.get("min_recording_ms", 1000))  # ignore silence cutoff before this

# Lazy engine singletons (instantiated on first use to keep startup fast).
_stt_engine: Optional[stt.STTEngine] = None
_tts_engine: Optional[tts.TTSEngine] = None
_llm_engine: Optional[llm.LLMEngine] = None
_wake_detector: Optional[wakeword.WakeWordDetector] = None
_command_router: Optional[commands.CommandRouter] = None


def _ensure_engines() -> None:
    global _stt_engine, _tts_engine, _llm_engine, _wake_detector, _command_router
    if _wake_detector is None:
        _wake_detector = wakeword.get_detector(CONFIG)
    if _stt_engine is None:
        _stt_engine = stt.get_engine(CONFIG)
    if _command_router is None:
        _command_router = commands.get_router(CONFIG)
    if CONFIG["enable_llm_response"] and _llm_engine is None:
        _llm_engine = llm.get_engine(CONFIG)
    if _tts_engine is None:
        _tts_engine = tts.get_engine(CONFIG)


_engines_ready = False
_engines_loading = False


async def _ensure_engines_async(session: "Session") -> None:
    """Load engines in a background thread, sending status pings to keep the WS alive."""
    global _engines_ready, _engines_loading
    if _engines_ready:
        return
    if _engines_loading:
        # Another task is already loading; wait for it.
        while _engines_loading:
            await session.send({"type": "status", "state": "loading"})
            await asyncio.sleep(2)
        return
    _engines_loading = True
    await session.send({"type": "status", "state": "loading"})
    loop = asyncio.get_event_loop()
    # Run the blocking init in a thread, but ping the client periodically.
    init_task = loop.run_in_executor(None, _ensure_engines)
    while not init_task.done():
        await session.send({"type": "status", "state": "loading"})
        await asyncio.sleep(3)
    try:
        init_task.result()
    except Exception as exc:
        logger.exception("Engine init failed")
        await session.send({"type": "error", "message": f"Engine init failed: {exc}"})
        raise
    _engines_ready = True
    _engines_loading = False


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "device": resolve_device(CONFIG),
        "config": {k: CONFIG[k] for k in ("wake_word", "stt_model", "tts_model", "llm_model")},
    }


@app.get("/config")
async def get_config() -> dict:
    return CONFIG


@app.post("/config")
async def update_config(patch: dict) -> dict:
    CONFIG.update(patch)
    return CONFIG


def _decode_audio(payload: str) -> np.ndarray:
    """Decode a base64-encoded float32 PCM buffer into a numpy array."""
    raw = base64.b64decode(payload)
    return np.frombuffer(raw, dtype=np.float32).copy()


def _encode_audio(audio: np.ndarray) -> str:
    """Encode a float32 numpy array as base64 PCM."""
    return base64.b64encode(audio.astype(np.float32).tobytes()).decode("ascii")


class Session:
    """Per-connection state machine for the assistant."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.state = "idle"
        self.buffer: list[np.ndarray] = []
        self.buffer_samples = 0
        self.silence_samples = 0
        self.recording_start: Optional[float] = None
        self.conversation: list[dict] = []
        self._lock = asyncio.Lock()
        self._heard_speech = False

    async def send(self, message: dict) -> None:
        try:
            await self.ws.send_json(message)
        except Exception as exc:
            logger.warning("Failed to send to client: %s", exc)

    async def set_state(self, state: str) -> None:
        self.state = state
        await self.send({"type": "status", "state": state})

    async def handle_audio(self, audio: np.ndarray) -> None:
        async with self._lock:
            if self.state == "idle":
                await self._detect_wake(audio)
            elif self.state == "listening":
                await self._accumulate_utterance(audio)

    async def force_listen(self) -> None:
        """Enter listening mode directly, interrupting any current state."""
        async with self._lock:
            logger.info("Force-listen triggered (was %s) -> listening", self.state)
            self.buffer = []
            self.buffer_samples = 0
            self.silence_samples = 0
            self._heard_speech = False
            self.recording_start = time.monotonic()
            await self.set_state("listening")

    async def _detect_wake(self, audio: np.ndarray) -> None:
        if _wake_detector is None:
            return
        # Run detector in a thread to avoid blocking the event loop.
        detected = await asyncio.to_thread(_wake_detector.detect, audio)
        if detected:
            logger.info("Wake word detected -> listening")
            self.buffer = [audio]
            self.buffer_samples = audio.size
            self.silence_samples = 0
            self._heard_speech = True  # wake word counts as speech
            self.recording_start = time.monotonic()
            await self.set_state("listening")

    async def _accumulate_utterance(self, audio: np.ndarray) -> None:
        self.buffer.append(audio)
        self.buffer_samples += audio.size
        rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0

        # Track whether we've heard actual speech in this utterance
        if rms >= SILENCE_THRESHOLD:
            self.silence_samples = 0
            self._heard_speech = True
        elif self._heard_speech:
            # Only count silence AFTER we've heard speech
            self.silence_samples += audio.size
        # else: still in initial silence, keep waiting for user to speak

        elapsed_ms = (time.monotonic() - (self.recording_start or time.monotonic())) * 1000
        silence_ms = (self.silence_samples / SAMPLE_RATE) * 1000

        # Don't cut off from silence until we've actually heard speech AND
        # enough minimum time has passed. Always respect max recording time.
        if elapsed_ms >= MAX_RECORDING_MS:
            logger.info("Max recording time reached (%dms), finalizing", elapsed_ms)
            await self._finalize_utterance()
        elif self._heard_speech and silence_ms >= SILENCE_DURATION_MS and elapsed_ms >= MIN_RECORDING_MS:
            logger.info("Silence detected (%dms after speech), finalizing", silence_ms)
            await self._finalize_utterance()
        elif not self._heard_speech and elapsed_ms >= MIN_RECORDING_MS * 3:
            # No speech heard at all after 3x min time — give up
            logger.info("No speech detected after %dms, cancelling", elapsed_ms)
            self.buffer = []
            self.buffer_samples = 0
            self.silence_samples = 0
            await self.set_state("idle")

    async def _finalize_utterance(self) -> None:
        if self.buffer_samples == 0:
            await self.set_state("idle")
            return
        full = np.concatenate(self.buffer)
        self.buffer = []
        self.buffer_samples = 0
        self.silence_samples = 0
        await self.set_state("transcribing")

        try:
            text = await asyncio.to_thread(_stt_engine.transcribe, full, SAMPLE_RATE)
        except Exception as exc:
            logger.exception("STT failed")
            await self.send({"type": "error", "message": f"STT failed: {exc}"})
            await self.set_state("idle")
            return

        if not text:
            logger.info("Empty transcript -> idle")
            await self.set_state("idle")
            return

        await self.send({"type": "transcript", "text": text})
        await self.set_state("thinking")
        await self._respond(text)

    async def _respond(self, text: str) -> None:
        response: Optional[str] = None
        if _command_router is not None:
            response = await asyncio.to_thread(_command_router.try_handle, text)
        if response is None:
            if _llm_engine is not None and CONFIG["enable_llm_response"]:
                history = list(self.conversation)
                response = await asyncio.to_thread(_llm_engine.respond, text, history)
            else:
                response = "I heard you, but LLM responses are disabled."
        if response is None:
            response = "I am not sure how to help with that."

        self.conversation.append({"role": "user", "content": text})
        self.conversation.append({"role": "assistant", "content": response})
        await self.send({"type": "response", "text": response})
        await self._speak(response)

    async def _speak(self, text: str) -> None:
        if _tts_engine is None:
            await self.set_state("idle")
            return
        await self.set_state("speaking")
        try:
            audio, out_sr = await asyncio.to_thread(_tts_engine.synthesize, text)
        except Exception as exc:
            logger.exception("TTS failed")
            await self.send({"type": "error", "message": f"TTS failed: {exc}"})
            await self.set_state("idle")
            return
        if audio.size:
            await self.send({"type": "audio", "data": _encode_audio(audio), "sample_rate": out_sr})
        await self.set_state("idle")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    await ws.send_json({"type": "ready", "sample_rate": SAMPLE_RATE})
    logger.info("Client connected; loading engines...")

    session = Session(ws)
    try:
        # Load engines in a background thread while keeping the WS alive.
        await _ensure_engines_async(session)
        await session.set_state("idle")
        logger.info("Engines ready")
    except Exception:
        await ws.close()
        return

    try:
        while True:
            raw = await ws.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await session.send({"type": "error", "message": "Invalid JSON"})
                continue
            mtype = message.get("type")
            if mtype == "audio":
                payload = message.get("data", "")
                if not payload:
                    continue
                audio = await asyncio.to_thread(_decode_audio, payload)
                await session.handle_audio(audio)
            elif mtype == "stop":
                session.buffer = []
                session.buffer_samples = 0
                await session.set_state("idle")
            elif mtype == "force_listen":
                await session.force_listen()
            elif mtype == "config":
                CONFIG.update(message.get("config", {}))
                await session.send({"type": "status", "state": session.state})
            else:
                await session.send({"type": "error", "message": f"Unknown type {mtype}"})
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as exc:
        logger.exception("WebSocket loop error")
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host=CONFIG["backend_host"],
        port=int(CONFIG["backend_port"]),
        reload=False,
    )
