from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doc_classifier.labels import DEFAULT_LABEL_SPACE
from doc_classifier.prompts import build_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3 1.7B for document classification.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--validation-file", type=Path, default=Path("data/processed/validation.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/qwen3-doc-classifier"))
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-input-chars", type=int, default=6000)
    parser.add_argument("--max-train-samples", type=int, default=0, help="Use 0 for all training rows.")
    parser.add_argument(
        "--max-validation-samples", type=int, default=0, help="Use 0 for all validation rows."
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    return parser.parse_args()


def tokenize_example(example, tokenizer, max_length: int, max_input_chars: int):
    label = DEFAULT_LABEL_SPACE.normalize(example["label"])
    prompt = build_prompt(example["text"], max_chars=max_input_chars)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(f" {label}{tokenizer.eos_token}", add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + answer_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def trainer_tokenizer_kwargs(tokenizer) -> dict:
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}


def main() -> None:
    args = parse_args()
    data_files = {"train": str(args.train_file)}
    if args.validation_file.exists():
        data_files["validation"] = str(args.validation_file)

    dataset = load_dataset("json", data_files=data_files)
    if args.max_train_samples > 0:
        dataset["train"] = dataset["train"].select(
            range(min(args.max_train_samples, len(dataset["train"])))
        )
    if "validation" in dataset and args.max_validation_samples > 0:
        dataset["validation"] = dataset["validation"].select(
            range(min(args.max_validation_samples, len(dataset["validation"])))
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenized = dataset.map(
        lambda example: tokenize_example(example, tokenizer, args.max_length, args.max_input_chars),
        remove_columns=dataset["train"].column_names,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        eval_strategy="steps" if "validation" in tokenized else "no",
        eval_steps=args.save_steps if "validation" in tokenized else None,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        **trainer_tokenizer_kwargs(tokenizer),
    )
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    (args.output_dir / "base_model_name.txt").write_text(args.model_name, encoding="utf-8")
    (args.output_dir / "labels.txt").write_text(
        "\n".join(DEFAULT_LABEL_SPACE.labels) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
