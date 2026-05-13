from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doc_classifier.extraction import extract_text
from doc_classifier.modeling import (
    classify_text,
    classify_text_strict,
    load_causal_lm,
    load_tokenizer,
    resolve_base_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify one document file.")
    parser.add_argument("--file", type=Path, help="Path to a text, PDF, or image document.")
    parser.add_argument("--text", help="Raw document text. Useful for quick tests.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter-path", type=Path, default=Path("models/qwen3-doc-classifier"))
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["score", "generate"],
        default="score",
        help="score always returns one allowed label. generate is the legacy free-text mode.",
    )
    parser.add_argument("--show-scores", action="store_true")
    parser.add_argument(
        "--label-batch-size",
        type=int,
        default=4,
        help="Number of candidate labels scored in one forward pass. Lower this if GPU memory is tight.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.file and not args.text:
        raise SystemExit("Provide --file or --text.")

    text = args.text if args.text is not None else extract_text(args.file)
    adapter = args.adapter_path if args.adapter_path.exists() else None
    base_model = resolve_base_model(args.adapter_path, args.model_name) if adapter else args.model_name
    tokenizer = load_tokenizer(str(adapter or base_model))
    model = load_causal_lm(base_model, str(adapter) if adapter else None, load_in_4bit=args.load_in_4bit)
    if args.mode == "score":
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
