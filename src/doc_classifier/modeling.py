"""Model loading and classification generation helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .labels import DEFAULT_LABEL_SPACE
from .prompts import build_prompt


def load_tokenizer(model_name_or_path: str):
    """Load a tokenizer and ensure it has a pad token for batching."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(
    model_name_or_path: str,
    adapter_path: str | None = None,
    load_in_4bit: bool = False,
):
    """Load the base causal LM and optionally attach a saved PEFT adapter."""
    quantization_config = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        quantization_config=quantization_config,
    )

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model


@torch.inference_mode()
def classify_text(
    text: str,
    model,
    tokenizer,
    max_input_chars: int = 6000,
    max_new_tokens: int = 16,
) -> tuple[str, str]:
    """Classify text with normal generation and parse only configured labels."""
    prompt = build_prompt(text, max_chars=max_input_chars)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    raw_answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    label = DEFAULT_LABEL_SPACE.label_from_generated_text(raw_answer)
    if label is None:
        label = DEFAULT_LABEL_SPACE.closest_from_text(raw_answer)
    return label, raw_answer


def resolve_base_model(adapter_path: Path, fallback: str) -> str:
    """Read the base model name saved with an adapter, or return the fallback."""
    config_path = adapter_path / "base_model_name.txt"
    if config_path.exists():
        return config_path.read_text(encoding="utf-8").strip()
    return fallback
