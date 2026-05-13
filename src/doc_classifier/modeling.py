"""Model loading and classification generation helpers."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .labels import DEFAULT_LABEL_SPACE
from .prompts import build_prompt


def load_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(
    model_name_or_path: str,
    adapter_path: str | None = None,
    load_in_4bit: bool = False,
):
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
def score_labels(
    text: str,
    model,
    tokenizer,
    max_input_chars: int = 6000,
) -> dict[str, float]:
    prompt = build_prompt(text, max_chars=max_input_chars)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    device = next(model.parameters()).device
    input_rows = []
    label_rows = []

    for label in DEFAULT_LABEL_SPACE.labels:
        label_ids = tokenizer(f" {label}{tokenizer.eos_token}", add_special_tokens=False)["input_ids"]
        input_rows.append(prompt_ids + label_ids)
        label_rows.append([-100] * len(prompt_ids) + label_ids)

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    max_length = max(len(row) for row in input_rows)
    input_ids = torch.full(
        (len(input_rows), max_length),
        fill_value=pad_token_id,
        dtype=torch.long,
        device=device,
    )
    labels = torch.full(
        (len(label_rows), max_length),
        fill_value=-100,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids, device=device)

    for row_index, (input_row, label_row) in enumerate(zip(input_rows, label_rows)):
        row_length = len(input_row)
        input_ids[row_index, :row_length] = torch.tensor(input_row, dtype=torch.long, device=device)
        labels[row_index, :row_length] = torch.tensor(label_row, dtype=torch.long, device=device)
        attention_mask[row_index, :row_length] = 1

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    loss_per_token = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(shifted_labels.shape)
    token_counts = (shifted_labels != -100).sum(dim=1).clamp_min(1)
    losses = loss_per_token.sum(dim=1) / token_counts

    scores = {
        label: float(loss.detach().cpu())
        for label, loss in zip(DEFAULT_LABEL_SPACE.labels, losses)
    }

    return scores


@torch.inference_mode()
def classify_text_by_score(
    text: str,
    model,
    tokenizer,
    max_input_chars: int = 6000,
) -> tuple[str, dict[str, float]]:
    scores = score_labels(text, model, tokenizer, max_input_chars=max_input_chars)
    label = min(scores, key=scores.get)
    return label, scores


@torch.inference_mode()
def classify_text(
    text: str,
    model,
    tokenizer,
    max_input_chars: int = 6000,
    max_new_tokens: int = 16,
) -> tuple[str, str]:
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
    label = DEFAULT_LABEL_SPACE.closest_from_text(raw_answer)
    return label, raw_answer


def classify_text_strict(
    text: str,
    model,
    tokenizer,
    max_input_chars: int = 6000,
) -> tuple[str, dict[str, float]]:
    label, scores = classify_text_by_score(
        text,
        model,
        tokenizer,
        max_input_chars=max_input_chars,
    )
    if not math.isfinite(scores[label]):
        raise RuntimeError("Could not score labels; best label has a non-finite score.")
    return label, scores


def resolve_base_model(adapter_path: Path, fallback: str) -> str:
    config_path = adapter_path / "base_model_name.txt"
    if config_path.exists():
        return config_path.read_text(encoding="utf-8").strip()
    return fallback
