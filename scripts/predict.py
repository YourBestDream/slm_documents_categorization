"""Classify one document or raw text using the fine-tuned Qwen adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doc_classifier.extraction import extract_text
from doc_classifier.modeling import (
    classify_text,
    classify_text_constrained,
    classify_text_hybrid,
    classify_text_strict,
    load_causal_lm,
    load_tokenizer,
    resolve_base_model,
)


def parse_args() -> argparse.Namespace:
    """Parse single-document prediction and model-loading arguments."""
    parser = argparse.ArgumentParser(description="Classify one document file.")
    parser.add_argument("--file", type=Path, help="Path to a text, PDF, or image document.")
    parser.add_argument("--text", help="Raw document text. Useful for quick tests.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter-path", type=Path, default=Path("models/qwen3-doc-classifier"))
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["constrained", "hybrid", "score", "generate"],
        default="generate",
        help="generate uses normal fine-tuned generation and parses only known labels.",
    )
    parser.add_argument("--show-scores", action="store_true")
    parser.add_argument(
        "--show-extracted-text",
        action="store_true",
        help="Print the extracted/OCR text before prediction.",
    )
    parser.add_argument(
        "--label-batch-size",
        type=int,
        default=4,
        help="Number of candidate labels scored in one forward pass. Lower this if GPU memory is tight.",
    )
    return parser.parse_args()


def main() -> None:
    """Extract document text when needed, load the model, and print the predicted category."""
    args = parse_args()
    if not args.file and not args.text:
        raise SystemExit("Provide --file or --text.")

    text = args.text if args.text is not None else extract_text(args.file)
    if args.show_extracted_text:
        print("extracted_text:")
        print(text)
        print("---")
    if args.adapter_path and not args.adapter_path.exists():
        raise FileNotFoundError(
            f"Adapter path does not exist: {args.adapter_path}. "
            "Restore the trained adapter or pass a valid --adapter-path."
        )
    adapter = args.adapter_path if args.adapter_path.exists() else None
    base_model = resolve_base_model(args.adapter_path, args.model_name) if adapter else args.model_name
    tokenizer = load_tokenizer(str(adapter or base_model))
    model = load_causal_lm(base_model, str(adapter) if adapter else None, load_in_4bit=args.load_in_4bit)
    if args.mode == "constrained":
        label, raw_answer = classify_text_constrained(text, model, tokenizer)
        print(f"category: {label}")
        if args.show_scores:
            print(json.dumps({"raw_answer": raw_answer, "mode": "constrained"}, indent=2))
    elif args.mode == "hybrid":
        label, raw_answer, scores = classify_text_hybrid(
            text,
            model,
            tokenizer,
            label_batch_size=args.label_batch_size,
        )
        print(f"category: {label}")
        if args.show_scores and scores is not None:
            print(json.dumps(dict(sorted(scores.items(), key=lambda item: item[1])), indent=2))
        elif args.show_scores:
            print(json.dumps({"raw_answer": raw_answer, "fallback_scoring_used": False}, indent=2))
    elif args.mode == "score":
        label, scores = classify_text_strict(
            text,
            model,
            tokenizer,
            label_batch_size=args.label_batch_size,
        )
        print(f"category: {label}")
        if args.show_scores:
            print(json.dumps(dict(sorted(scores.items(), key=lambda item: item[1])), indent=2))
    else:
        label, raw_answer = classify_text(text, model, tokenizer)
        print(f"category: {label}")
        print(f"raw_answer: {raw_answer}")


if __name__ == "__main__":
    main()
