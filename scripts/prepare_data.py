from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

from datasets import load_dataset
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doc_classifier.io import read_jsonl, write_jsonl
from doc_classifier.labels import DEFAULT_LABEL_SPACE, RVL_CDIP_LABELS


DEFAULT_SPLIT_MAP: dict[str, str] = {
    "train": "train",
    "validation": "val",
    "test": "test",
}

SPLIT_SEED_OFFSET: dict[str, int] = {
    "train": 0,
    "validation": 1,
    "test": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare OCR text JSONL files from an open-source document dataset."
    )
    parser.add_argument(
        "--dataset-name",
        default="chainyo/rvl-cdip",
        help="Parquet-backed RVL-CDIP mirror. Avoids legacy HF dataset scripts.",
    )
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming and download/cache dataset shards locally.",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=250,
        help="Use 0 to process the full split. Ignored when --samples-per-label is set.",
    )
    parser.add_argument(
        "--samples-per-label",
        type=int,
        default=0,
        help="Write up to this many OCR-valid examples per label for each split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=10000,
        help="Streaming shuffle buffer. Use 0 to disable shuffling.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress after this many written records. Use 0 to disable.",
    )
    parser.add_argument("--min-text-chars", type=int, default=20)
    return parser.parse_args()


def label_to_name(dataset_split, label_column: str, value: Any) -> str:
    feature = dataset_split.features.get(label_column)
    if hasattr(feature, "int2str"):
        return DEFAULT_LABEL_SPACE.normalize(feature.int2str(value))
    if isinstance(value, int):
        return DEFAULT_LABEL_SPACE.normalize(RVL_CDIP_LABELS[value])
    return DEFAULT_LABEL_SPACE.normalize(str(value))


def ocr_image(image) -> str:
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError("Install pytesseract to OCR document images.") from exc

    try:
        if isinstance(image, dict):
            if image.get("bytes") is not None:
                image = Image.open(BytesIO(image["bytes"]))
            elif image.get("path") is not None:
                image = Image.open(image["path"])
        return pytesseract.image_to_string(image).strip()
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract OCR executable was not found. Install Tesseract and ensure it is on PATH."
        ) from exc


def dataset_split_name(output_split: str) -> str:
    return DEFAULT_SPLIT_MAP.get(output_split, output_split)


def split_seed(args: argparse.Namespace, output_split: str) -> int:
    return args.seed + SPLIT_SEED_OFFSET.get(output_split, 0)


def maybe_shuffle_dataset(dataset_split, args: argparse.Namespace, output_split: str):
    if args.shuffle_buffer_size <= 0:
        return dataset_split
    if args.no_streaming:
        return dataset_split.shuffle(seed=split_seed(args, output_split))
    return dataset_split.shuffle(
        seed=split_seed(args, output_split),
        buffer_size=args.shuffle_buffer_size,
    )


def target_records(args: argparse.Namespace) -> int | None:
    if args.samples_per_label > 0:
        return args.samples_per_label * len(DEFAULT_LABEL_SPACE.labels)
    if args.max_samples_per_split > 0:
        return args.max_samples_per_split
    return None


def print_progress(
    output_split: str,
    written: int,
    seen: int,
    target: int | None,
    started_at: float,
    label_counts: Counter[str],
) -> None:
    elapsed = time.monotonic() - started_at
    rate = written / elapsed if elapsed > 0 else 0.0
    percent = f"{(written / target) * 100:5.1f}%" if target else "  n/a"
    target_text = str(target) if target else "unknown"
    labels_text = ", ".join(
        f"{label}={count}" for label, count in sorted(label_counts.items()) if count
    )
    print(
        f"[{output_split}] {written}/{target_text} written ({percent}); "
        f"{seen} rows scanned; {rate:.2f} records/s; elapsed {elapsed / 60:.1f} min; "
        f"labels: {labels_text}",
        flush=True,
    )


def iter_records(args: argparse.Namespace, output_split: str):
    source_split = dataset_split_name(output_split)
    dataset_split = load_dataset(
        args.dataset_name,
        split=source_split,
        streaming=not args.no_streaming,
    )
    dataset_split = maybe_shuffle_dataset(dataset_split, args, output_split)

    if args.samples_per_label <= 0 and args.max_samples_per_split > 0 and args.no_streaming:
        dataset_split = dataset_split.select(range(min(args.max_samples_per_split, len(dataset_split))))

    label_counts: Counter[str] = Counter()
    total_written = 0
    started_at = time.monotonic()
    target = target_records(args)
    for index, example in enumerate(dataset_split):
        label = label_to_name(dataset_split, args.label_column, example[args.label_column])
        if args.samples_per_label > 0 and label_counts[label] >= args.samples_per_label:
            if all(label_counts[item] >= args.samples_per_label for item in DEFAULT_LABEL_SPACE.labels):
                break
            continue

        if (
            args.samples_per_label <= 0
            and args.max_samples_per_split > 0
            and total_written >= args.max_samples_per_split
        ):
            break

        text = ocr_image(example[args.image_column])
        if len(text) < args.min_text_chars:
            continue

        label_counts[label] += 1
        total_written += 1
        if args.progress_every > 0 and total_written % args.progress_every == 0:
            print_progress(
                output_split,
                written=total_written,
                seen=index + 1,
                target=target,
                started_at=started_at,
                label_counts=label_counts,
            )
        yield {
            "id": f"{output_split}-{index}",
            "source_dataset": args.dataset_name,
            "split": output_split,
            "text": text,
            "label": label,
        }

    print_progress(
        output_split,
        written=total_written,
        seen=index + 1 if "index" in locals() else 0,
        target=target,
        started_at=started_at,
        label_counts=label_counts,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for output_split in args.splits:
        output_path = args.output_dir / f"{output_split}.jsonl"
        count = write_jsonl(output_path, iter_records(args, output_split))
        print(f"Wrote {count} records to {output_path}")
        distribution = Counter(record["label"] for record in read_jsonl(output_path))
        print(f"Label distribution: {dict(sorted(distribution.items()))}")


if __name__ == "__main__":
    main()
