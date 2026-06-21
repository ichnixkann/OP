"""Wake word detection using openWakeWord.

Falls back to a simple energy-based detector if openWakeWord is unavailable
so the assistant remains usable while models download.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """Detect a configurable wake word from a stream of PCM audio chunks."""

    def __init__(self, wake_word: str, sensitivity: float, sample_rate: int):
        self.wake_word = wake_word.lower().strip()
        self.sensitivity = float(sensitivity)
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self._model = None
        self._fallback = False
        self._init_model()

    def _init_model(self) -> None:
        try:
            from openwakeword import Model
            from openwakeword.utils import download_models

            download_models()  # idempotent cache check
            # openWakeWord ships a small set of preset models. We try to match
            # the configured wake word to one of them; otherwise we use the
            # generic "hey_jarvis" model as a stand-in and rely on the user
            # editing config to pick a supported name.
            available = {
                "hey jarvis": "hey_jarvis",
                "alexa": "alexa",
                "hey mycroft": "hey_mycroft",
                "computer": "hey_jarvis",  # closest preset
                "hey computer": "hey_jarvis",
            }
            model_name = available.get(self.wake_word, "hey_jarvis")
            logger.info("Loading openWakeWord model %s for wake word %r", model_name, self.wake_word)
            # Prefer onnxruntime to avoid numpy 2.x compat issues with tflite_runtime.
            try:
                self._model = Model(wakeword_models=[model_name], inference_framework="onnx")
            except TypeError:
                # Older openWakeWord versions don't accept inference_framework.
                self._model = Model(wakeword_models=[model_name])
            logger.info("openWakeWord ready")
        except Exception as exc:
            logger.warning("openWakeWord unavailable (%s); using energy fallback", exc)
            self._fallback = True

    def detect(self, audio: np.ndarray) -> bool:
        """Return True if the wake word is detected in this chunk."""
        if audio.size == 0:
            return False
        if audio.dtype != np.int16:
            # openWakeWord expects int16 PCM.
            audio_i16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        else:
            audio_i16 = audio

        with self._lock:
            if self._fallback:
                return self._energy_detect(audio_i16)
            try:
                scores = self._model.predict(audio_i16)
                # scores is a dict {model_name: score}
                for name, score in scores.items():
                    if score >= self.sensitivity:
                        logger.info("Wake word detected (score=%.2f, model=%s)", score, name)
                        return True
                return False
            except Exception as exc:
                logger.warning("openWakeWord predict failed: %s", exc)
                return self._energy_detect(audio_i16)

    def _energy_detect(self, audio_i16: np.ndarray) -> bool:
        """Fallback: trigger on speech-level audio energy.

        Normal speech has RMS ~0.05-0.15; silence is ~0.005-0.02.
        We use a configurable threshold (default 0.06) so any spoken
        phrase triggers listening — matching the 'always listening' mode.
        """
        rms = float(np.sqrt(np.mean((audio_i16.astype(np.float32) / 32767.0) ** 2)))
        triggered = rms > self.sensitivity
        if triggered:
            logger.info("Energy-based wake trigger (rms=%.3f, threshold=%.3f)", rms, self.sensitivity)
        return triggered


_detector: Optional[WakeWordDetector] = None


def get_detector(config: dict) -> WakeWordDetector:
    global _detector
    if _detector is None:
        _detector = WakeWordDetector(
            wake_word=config["wake_word"],
            sensitivity=config["wake_word_sensitivity"],
            sample_rate=config["sample_rate"],
        )
    return _detector
