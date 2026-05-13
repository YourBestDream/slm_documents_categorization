from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doc_classifier.io import read_jsonl
from doc_classifier.labels import DEFAULT_LABEL_SPACE
from doc_classifier.modeling import (
    classify_text,
    classify_text_strict,
    load_causal_lm,
    load_tokenizer,
    resolve_base_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the document classifier.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter-path", type=Path, default=Path("models/qwen3-doc-classifier"))
    parser.add_argument("--test-file", type=Path, default=Path("data/processed/test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--max-samples", type=int, default=0, help="Use 0 for all examples.")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print evaluation progress after this many records. Use 0 to disable.",
    )
    parser.add_argument(
        "--label-batch-size",
        type=int,
        default=4,
        help="Number of candidate labels scored in one forward pass. Lower this if GPU memory is tight.",
    )
    parser.add_argument(
        "--mode",
        choices=["score", "generate"],
        default="score",
        help="score always chooses one allowed label. generate is the legacy free-text mode.",
    )
    return parser.parse_args()


def count_records(path: Path, max_samples: int = 0) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as file:
        for count, _ in enumerate(file, start=1):
            if max_samples and count >= max_samples:
                return count
    return count


def print_progress(completed: int, total: int, started_at: float) -> None:
    elapsed = time.monotonic() - started_at
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = (total - completed) / rate if rate > 0 else 0.0
    percent = (completed / total) * 100 if total else 0.0
    print(
        f"[evaluate] {completed}/{total} ({percent:5.1f}%); "
        f"{rate:.2f} docs/s; elapsed {elapsed / 60:.1f} min; "
        f"eta {remaining / 60:.1f} min",
        flush=True,
    )


def save_confusion_matrix(y_true: list[str], y_pred: list[str], output_path: Path) -> None:
    labels = list(DEFAULT_LABEL_SPACE.labels)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adapter = args.adapter_path if args.adapter_path.exists() else None
    base_model = resolve_base_model(args.adapter_path, args.model_name) if adapter else args.model_name
    tokenizer = load_tokenizer(str(adapter or base_model))
    model = load_causal_lm(base_model, str(adapter) if adapter else None, load_in_4bit=args.load_in_4bit)

    y_true: list[str] = []
    y_pred: list[str] = []
    predictions = []
    total_records = count_records(args.test_file, args.max_samples)
    started_at = time.monotonic()
    for index, record in enumerate(read_jsonl(args.test_file)):
        if args.max_samples and index >= args.max_samples:
            break
        expected = DEFAULT_LABEL_SPACE.normalize(record["label"])
        if args.mode == "score":
            predicted, scores = classify_text_strict(
                record["text"],
                model,
                tokenizer,
                label_batch_size=args.label_batch_size,
            )
            raw_answer = predicted
            score_payload = scores
        else:
            predicted, raw_answer = classify_text(record["text"], model, tokenizer)
            score_payload = None
        y_true.append(expected)
        y_pred.append(predicted)
        prediction_record = {
            "id": record.get("id", str(index)),
            "label": expected,
            "prediction": predicted,
            "raw_answer": raw_answer,
        }
        if score_payload is not None:
            prediction_record["label_scores"] = score_payload
        predictions.append(prediction_record)
        completed = index + 1
        if args.progress_every > 0 and completed % args.progress_every == 0:
            print_progress(completed, total_records, started_at)

    if y_true:
        print_progress(len(y_true), total_records, started_at)

    metrics = {
        "total": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred) if y_true else 0.0,
        "macro_f1": f1_score(y_true, y_pred, labels=list(DEFAULT_LABEL_SPACE.labels), average="macro")
        if y_true
        else 0.0,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(DEFAULT_LABEL_SPACE.labels),
            zero_division=0,
            output_dict=True,
        )
        if y_true
        else {},
    }

    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "predictions.json").write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if y_true:
        save_confusion_matrix(y_true, y_pred, args.output_dir / "confusion_matrix.png")

    print(json.dumps({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
