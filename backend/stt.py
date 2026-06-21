"""Speech-to-Text using Distil-Whisper via HuggingFace transformers."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class STTEngine:
    """Wraps a Distil-Whisper pipeline for transcription of PCM audio."""

    def __init__(self, model_name: str, language: str, device: str):
        from transformers import pipeline

        logger.info("Loading STT model %s on %s", model_name, device)
        self.device = device
        self.language = language
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device,
            chunk_length_s=30,
        )
        logger.info("STT model loaded")

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe a mono float32 numpy array of audio samples."""
        if audio.size == 0:
            return ""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        # Whisper pipelines expect float32 in [-1, 1].
        peak = float(np.max(np.abs(audio))) or 1.0
        if peak > 1.0:
            audio = audio / peak
        result = self._pipe(
            {"array": audio, "sampling_rate": sample_rate},
            generate_kwargs={"language": self.language, "task": "transcribe"},
        )
        text = (result.get("text") or "").strip()
        logger.info("Transcribed: %r", text)
        return text


_engine: Optional[STTEngine] = None


def get_engine(config: dict) -> STTEngine:
    global _engine
    if _engine is None:
        from .config import resolve_device

        device = resolve_device(config)
        _engine = STTEngine(
            model_name=config["stt_model"],
            language=config["stt_language"],
            device=device,
        )
    return _engine
