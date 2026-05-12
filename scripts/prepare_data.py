from __future__ import annotations

import argparse
import sys
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
        yield {
            "id": f"{output_split}-{index}",
            "source_dataset": args.dataset_name,
            "split": output_split,
            "text": text,
            "label": label,
        }


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
