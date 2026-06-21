"""Local LLM response generation using a small HuggingFace instruct model."""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LLMEngine:
    """Wraps a small instruct-tuned HF model for response generation."""

    SYSTEM_PROMPT = (
        "You are a helpful, concise desktop voice assistant. "
        "Answer in one or two short sentences unless the user asks for detail. "
        "If the user asks to perform an action that you cannot do directly, "
        "say so plainly."
    )

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int,
        temperature: float,
        device: str,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading LLM %s on %s", model_name, device)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(device)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        logger.info("LLM loaded")

    def respond(self, user_text: str, history: Optional[list[dict]] = None) -> str:
        """Generate a response given user text and optional prior conversation."""
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        if history:
            messages.extend(history[-6:])
        messages.append({"role": "user", "content": user_text})

        try:
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback for tokenizers without chat template support.
            prompt = "\n".join(
                f"[{m['role'].upper()}] {m['content']}" for m in messages
            ) + "\n[ASSISTANT] "

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        import torch

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-2),
                pad_token_id=self._tokenizer.pad_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        logger.info("LLM response: %r", text)
        return text


_engine: Optional[LLMEngine] = None


def get_engine(config: dict) -> LLMEngine:
    global _engine
    if _engine is None:
        from .config import resolve_device

        device = resolve_device(config)
        _engine = LLMEngine(
            model_name=config["llm_model"],
            max_new_tokens=config["llm_max_new_tokens"],
            temperature=config["llm_temperature"],
            device=device,
        )
    return _engine
