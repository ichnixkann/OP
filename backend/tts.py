"""Text-to-Speech engine supporting multiple model architectures.

Supported model types:
- VITS/MMS models (e.g. facebook/mms-tts-eng) — natural, fast, no speaker embeddings needed
- SpeechT5 (microsoft/speecht5_tts) — configurable speaker via embeddings dataset
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _is_vits_model(model_name: str) -> bool:
    """Detect VITS/MMS model by name pattern."""
    lower = model_name.lower()
    return "mms-tts" in lower or "vits" in lower


class VITSEngine:
    """VITS/MMS TTS — produces natural speech without speaker embeddings."""

    def __init__(self, model_name: str, device: str):
        import torch
        from transformers import VitsModel, AutoTokenizer

        logger.info("Loading VITS TTS model %s on %s", model_name, device)
        self.device = device
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = VitsModel.from_pretrained(model_name).to(device)
        self._sample_rate = self._model.config.sampling_rate
        logger.info("VITS TTS model loaded (sample_rate=%d)", self._sample_rate)

    def synthesize(self, text: str, sample_rate: int = 22050) -> tuple[np.ndarray, int]:
        """Synthesize text to PCM audio. Returns (float32 mono, output_sample_rate)."""
        import torch

        text = (text or "").strip()
        if not text:
            return np.zeros(0, dtype=np.float32), self._sample_rate

        inputs = self._tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self._model(**inputs)
        audio = output.waveform[0].cpu().numpy().astype(np.float32)
        logger.info("Synthesized %d samples (%.2fs)", audio.size, audio.size / self._sample_rate)
        return audio, self._sample_rate


class SpeechT5Engine:
    """SpeechT5 TTS with configurable speaker embeddings."""

    def __init__(self, model_name: str, speaker_embeddings_name: str, device: str):
        import torch
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

        logger.info("Loading SpeechT5 TTS model %s on %s", model_name, device)
        self.device = device
        self._processor = SpeechT5Processor.from_pretrained(model_name)
        self._model = SpeechT5ForTextToSpeech.from_pretrained(model_name).to(device)
        self._vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
        self._speaker_embeddings = self._load_speaker_embeddings(speaker_embeddings_name, device)
        logger.info("SpeechT5 TTS model loaded")

    def _load_speaker_embeddings(self, name: str, device: str):
        """Try multiple sources for speaker embeddings."""
        import torch

        try:
            from datasets import load_dataset

            ds = load_dataset(name, split="validation")
            emb = torch.tensor(ds[7306]["xvector"]).unsqueeze(0).to(device)
            logger.info("Speaker embeddings loaded from dataset %s", name)
            return emb
        except Exception as exc:
            logger.info("Dataset %s unavailable (%s), trying alternatives…", name, exc)

        try:
            from datasets import load_dataset

            ds = load_dataset(name, split="train")
            emb = torch.tensor(ds[0]["xvector"]).unsqueeze(0).to(device)
            logger.info("Speaker embeddings loaded from dataset %s (train split)", name)
            return emb
        except Exception as exc:
            logger.info("Dataset %s train split also unavailable: %s", name, exc)

        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo_id="microsoft/speecht5_tts", filename="speaker_embeddings.pt")
            emb = torch.load(path, map_location=device, weights_only=True)
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            logger.info("Speaker embeddings loaded from speecht5_tts repo")
            return emb.to(device)
        except Exception as exc:
            logger.info("hf_hub_download for speaker_embeddings.pt failed: %s", exc)

        torch.manual_seed(42)
        emb = torch.randn(1, 512) * 0.1
        emb = emb / emb.norm() * 10.0
        logger.warning("Using deterministic random speaker embeddings (voice may vary)")
        return emb.to(device)

    def synthesize(self, text: str, sample_rate: int = 22050) -> tuple[np.ndarray, int]:
        """Synthesize text to PCM audio. Returns (float32 mono, output_sample_rate)."""
        import torch

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
        out_sr = 22050
        logger.info("Synthesized %d samples (%.2fs)", audio.size, audio.size / out_sr)
        return audio, out_sr


import torch  # noqa: E402

_engine: Optional[VITSEngine | SpeechT5Engine] = None


def get_engine(config: dict) -> VITSEngine | SpeechT5Engine:
    global _engine
    if _engine is None:
        from .config import resolve_device

        device = resolve_device(config)
        model_name = config["tts_model"]

        if _is_vits_model(model_name):
            _engine = VITSEngine(model_name=model_name, device=device)
        else:
            _engine = SpeechT5Engine(
                model_name=model_name,
                speaker_embeddings_name=config["tts_speaker_embeddings"],
                device=device,
            )
    return _engine
