"""Shared configuration loader for the voice assistant backend."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_DEFAULT_CONFIG: dict[str, Any] = {
    "wake_word": "computer",
    "wake_word_sensitivity": 0.06,
    "stt_model": "distil-whisper/distil-large-v3",
    "stt_language": "en",
    "tts_model": "microsoft/speecht5_tts",
    "tts_speaker_embeddings": "Matthijs/cmu_xim2",
    "llm_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "llm_max_new_tokens": 256,
    "llm_temperature": 0.7,
    "device": "auto",
    "sample_rate": 16000,
    "silence_threshold": 0.01,
    "silence_duration_ms": 800,
    "max_recording_ms": 15000,
    "enable_command_execution": True,
    "enable_llm_response": True,
    "backend_host": "127.0.0.1",
    "backend_port": 8765,
}


def _config_path() -> Path:
    """Resolve config.json location: project root, then backend dir, then env override."""
    env_path = os.environ.get("VOICE_ASSISTANT_CONFIG")
    if env_path:
        return Path(env_path)
    here = Path(__file__).resolve().parent
    candidates = [here.parent / "config.json", here / "config.json"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_config() -> dict[str, Any]:
    """Load merged config (defaults + config.json overrides)."""
    config = dict(_DEFAULT_CONFIG)
    path = _config_path()
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            user_cfg = json.load(fh)
        config.update(user_cfg)
    return config


def resolve_device(config: dict[str, Any]) -> str:
    """Resolve 'auto' device to 'cuda' if available, else 'cpu'."""
    device = config.get("device", "auto")
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device
