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

        # Load speaker embeddings. The Matthijs/cmu_xim2 dataset requires the
        # `datasets` library and has been unreliable on the Hub. Instead we
        # download the speaker embedding vector directly as a .pt file from
        # the SpeechT5 model card examples, or generate a deterministic one.
        self._speaker_embeddings = self._load_speaker_embeddings(speaker_embeddings_name, device)
        logger.info("TTS model loaded")

    def _load_speaker_embeddings(self, name: str, device: str):
        """Try multiple sources for speaker embeddings; never fall back to zeros."""
        import torch

        # Source 1: try the datasets library (original approach)
        try:
            from datasets import load_dataset  # type: ignore

            ds = load_dataset(name, split="train")
            emb = torch.tensor(ds[0]["xvector"]).unsqueeze(0).to(device)
            logger.info("Speaker embeddings loaded from dataset %s", name)
            return emb
        except Exception as exc:
            logger.info("Dataset %s unavailable (%s), trying alternatives…", name, exc)

        # Source 2: download a pre-computed speaker embedding .pt file
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo_id="microsoft/speecht5_tts", filename="speaker_embeddings.pt")
            emb = torch.load(path, map_location=device).unsqueeze(0).to(device)
            logger.info("Speaker embeddings loaded from speecht5_tts repo")
            return emb
        except Exception as exc:
            logger.info("hf_hub_download for speaker_embeddings.pt failed: %s", exc)

        # Source 3: generate a deterministic embedding (not zeros — zeros produce
        # garbage audio). Use a fixed seed so the voice is consistent across runs.
        torch.manual_seed(42)
        emb = torch.randn(1, 512) * 0.1
        emb = emb / emb.norm() * 10.0  # normalise to typical xvector magnitude
        logger.warning("Using deterministic random speaker embeddings (voice may vary)")
        return emb.to(device)

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
