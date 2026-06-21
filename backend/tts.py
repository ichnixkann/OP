"""Text-to-Speech using Microsoft SpeechT5 via HuggingFace transformers."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class TTSEngine:
    """Wraps SpeechT5 for synthesising speech from text."""

    def __init__(
        self,
        model_name: str,
        speaker_embeddings_name: str,
        device: str,
    ):
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
        from transformers import AutoProcessor

        logger.info("Loading TTS model %s on %s", model_name, device)
        self.device = device
        self._processor = SpeechT5Processor.from_pretrained(model_name)
        self._model = SpeechT5ForTextToSpeech.from_pretrained(model_name).to(device)
        self._vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

        # Load speaker embeddings (XIM2 voice).
        from datasets import load_dataset  # type: ignore

        try:
            emb_dataset = load_dataset(speaker_embeddings_name, split="train")
            self._speaker_embeddings = torch.tensor(emb_dataset[0]["xvector"]).unsqueeze(0).to(device)
        except Exception as exc:
            logger.warning("Falling back to default speaker embeddings: %s", exc)
            self._speaker_embeddings = torch.zeros(1, 512).to(device)
        logger.info("TTS model loaded")

    def synthesize(self, text: str, sample_rate: int = 22050) -> tuple[np.ndarray, int]:
        """Synthesize text to PCM audio. Returns (float32 mono, output_sample_rate)."""
        text = (text or "").strip()
        if not text:
            return np.zeros(0, dtype=np.float32), sample_rate
        inputs = self._processor(text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            waveform = self._model.generate_speech(
                inputs["input_ids"],
                self._speaker_embeddings,
                vocoder=self._vocoder,
            )
        audio = waveform.cpu().numpy().astype(np.float32)
        out_sr = 22050  # SpeechT5 native rate
        logger.info("Synthesized %d samples (%.2fs)", audio.size, audio.size / out_sr)
        return audio, out_sr


import torch  # noqa: E402  (kept at bottom to keep top-level imports tidy)

_engine: Optional[TTSEngine] = None


def get_engine(config: dict) -> TTSEngine:
    global _engine
    if _engine is None:
        from .config import resolve_device

        device = resolve_device(config)
        _engine = TTSEngine(
            model_name=config["tts_model"],
            speaker_embeddings_name=config["tts_speaker_embeddings"],
            device=device,
        )
    return _engine
