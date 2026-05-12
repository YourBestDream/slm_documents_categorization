from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doc_classifier.io import write_jsonl
from doc_classifier.labels import DEFAULT_LABEL_SPACE, RVL_CDIP_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare OCR text JSONL files from an open-source document dataset."
    )
    parser.add_argument("--dataset-name", default="aharley/rvl_cdip")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=250,
        help="Use 0 to process the full split. The default is intentionally small.",
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
        return pytesseract.image_to_string(image).strip()
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract OCR executable was not found. Install Tesseract and ensure it is on PATH."
        ) from exc


def iter_records(args: argparse.Namespace, split: str):
    dataset_split = load_dataset(args.dataset_name, split=split)
    if args.max_samples_per_split > 0:
        dataset_split = dataset_split.select(range(min(args.max_samples_per_split, len(dataset_split))))

    for index, example in enumerate(dataset_split):
        text = ocr_image(example[args.image_column])
        if len(text) < args.min_text_chars:
            continue
        yield {
            "id": f"{split}-{index}",
            "source_dataset": args.dataset_name,
            "split": split,
            "text": text,
            "label": label_to_name(dataset_split, args.label_column, example[args.label_column]),
        }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        output_path = args.output_dir / f"{split}.jsonl"
        count = write_jsonl(output_path, iter_records(args, split))
        print(f"Wrote {count} records to {output_path}")


if __name__ == "__main__":
    main()

