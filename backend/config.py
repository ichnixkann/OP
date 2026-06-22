"""Shared configuration loader for the voice assistant backend."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG: dict[str, Any] = {
    "wake_word": "computer",
    "wake_word_sensitivity": 0.06,
    "stt_model": "distil-whisper/distil-large-v3",
    "stt_language": "en",
    "tts_model": "microsoft/speecht5_tts",
    "tts_speaker_embeddings": "Matthijs/cmu-arctic-xvectors",
    "llm_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "llm_max_new_tokens": 256,
    "llm_temperature": 0.7,
    "device": "auto",
    "sample_rate": 16000,
    "silence_threshold": 0.02,
    "silence_duration_ms": 1200,
    "min_recording_ms": 1000,
    "max_recording_ms": 15000,
    "enable_command_execution": True,
    "enable_llm_response": True,
    "backend_host": "127.0.0.1",
    "backend_port": 8765,
}

# Validation constraints for numeric config values.
_VALIDATORS: dict[str, tuple[float, float]] = {
    "wake_word_sensitivity": (0.01, 0.5),
    "llm_max_new_tokens": (16, 2048),
    "llm_temperature": (0.0, 2.0),
    "sample_rate": (8000, 48000),
    "silence_threshold": (0.001, 0.5),
    "silence_duration_ms": (200, 5000),
    "min_recording_ms": (200, 10000),
    "max_recording_ms": (2000, 60000),
    "backend_port": (1, 65535),
}

# Keys that are safe to update at runtime (no engine restart needed).
SAFE_RUNTIME_KEYS = {
    "wake_word",
    "wake_word_sensitivity",
    "silence_threshold",
    "silence_duration_ms",
    "min_recording_ms",
    "max_recording_ms",
    "enable_command_execution",
    "enable_llm_response",
    "llm_max_new_tokens",
    "llm_temperature",
}

# Keys that require an engine restart to take effect.
RESTART_KEYS = {
    "stt_model",
    "stt_language",
    "tts_model",
    "tts_speaker_embeddings",
    "llm_model",
    "device",
    "sample_rate",
    "backend_host",
    "backend_port",
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


def save_config(config: dict[str, Any]) -> None:
    """Persist current config to config.json on disk."""
    path = _config_path()
    # Only persist keys that differ from defaults or are user-set.
    to_save = {k: v for k, v in config.items() if k in _DEFAULT_CONFIG}
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(to_save, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        logger.info("Config saved to %s", path)
    except OSError as exc:
        logger.error("Failed to save config: %s", exc)


def validate_config(patch: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Validate and coerce a config patch. Returns (cleaned_patch, errors)."""
    errors: list[str] = []
    cleaned: dict[str, Any] = {}

    for key, value in patch.items():
        if key not in _DEFAULT_CONFIG:
            errors.append(f"Unknown config key: {key}")
            continue

        expected_type = type(_DEFAULT_CONFIG[key])

        # Coerce types
        try:
            if expected_type is bool:
                if isinstance(value, str):
                    value = value.lower() in ("true", "1", "yes")
                else:
                    value = bool(value)
            elif expected_type is int:
                value = int(float(value))
            elif expected_type is float:
                value = float(value)
            elif expected_type is str:
                value = str(value).strip()
        except (ValueError, TypeError):
            errors.append(f"{key}: cannot convert {value!r} to {expected_type.__name__}")
            continue

        # Range validation
        if key in _VALIDATORS:
            lo, hi = _VALIDATORS[key]
            if not (lo <= value <= hi):
                errors.append(f"{key}: value {value} out of range [{lo}, {hi}]")
                continue

        # String validation
        if expected_type is str and not value:
            errors.append(f"{key}: cannot be empty")
            continue

        cleaned[key] = value

    return cleaned, errors


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
