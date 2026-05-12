# SLM Documents Categorization

This project fine-tunes **Qwen3 1.7B** with LoRA for document category classification.
The default dataset pipeline uses a Parquet-backed mirror of the open-source **RVL-CDIP**
document classification dataset, extracts OCR text from each document image, and trains
the model to return one category for each document.

## Labels

The default label space follows RVL-CDIP:

`letter`, `form`, `email`, `handwritten`, `advertisement`, `scientific report`,
`scientific publication`, `specification`, `file folder`, `news article`, `budget`,
`invoice`, `presentation`, `questionnaire`, `resume`, `memo`.

Business categories such as `banking` are not a native RVL-CDIP class. This implementation
maps common banking terms to `budget` as a pragmatic fallback. For production banking
classification, add banking statement examples and extend `src/doc_classifier/labels.py`.

## Docker Setup

Docker is the recommended setup for this project because the image includes Tesseract OCR.
You do not need to install Tesseract on your host machine.

Requirements:

- Docker Desktop
- Docker Compose v2
- Optional: `make`

Build the image:

```bash
docker compose build
```

Or with Make:

```bash
make docker-build
```

Open a shell in the container:

```bash
docker compose run --rm app bash
```

Or:

```bash
make shell
```

The project directory is mounted into the container, so generated files appear on your host
under `data/processed`, `models`, and `outputs`. Hugging Face downloads are stored in a
Docker volume named `hf-cache`, so model and dataset downloads are reused between runs.

This Docker setup is CPU-only. It is suitable for OCR, data preparation, prediction, and
small smoke tests. Full fine-tuning of Qwen3 1.7B on CPU can be very slow and may require
more RAM than Docker Desktop gives the container by default.

## Local Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

For OCR from document images, install the Tesseract executable and ensure it is available
on `PATH`. The Python package `pytesseract` is installed by the project dependencies, but
it still needs the native Tesseract program.

Skip local setup if you use Docker.

## Prepare Data

The default preparation command processes a small sample from each split so the pipeline
can be tested quickly:

```bash
python scripts/prepare_data.py
```

The default dataset is `chainyo/rvl-cdip`, a Parquet-backed RVL-CDIP mirror. This avoids
legacy Hugging Face dataset scripts that recent `datasets` versions no longer load.
Preparation uses streaming by default, so sample runs do not need to download the full
dataset first.

With Docker and Make:

```bash
make prepare
```

Without Make:

```bash
docker compose run --rm app python scripts/prepare_data.py --max-samples-per-split 25
```

To process the full RVL-CDIP splits, use:

```bash
python scripts/prepare_data.py --max-samples-per-split 0
```

With Docker and Make:

```bash
make prepare-full
```

Prepared files are written to:

- `data/processed/train.jsonl`
- `data/processed/validation.jsonl`
- `data/processed/test.jsonl`

Each JSONL row has this shape:

```json
{"id": "train-0", "source_dataset": "chainyo/rvl-cdip", "split": "train", "text": "...", "label": "invoice"}
```

## Fine-Tune

Run LoRA fine-tuning:

```bash
python scripts/train.py
```

For GPU environments that support `bitsandbytes`, QLoRA can reduce memory use:

```bash
python scripts/train.py --load-in-4bit
```

The adapter is saved to:

```text
models/qwen3-doc-classifier/
```

Useful training flags:

```bash
python scripts/train.py --epochs 3 --batch-size 2 --gradient-accumulation-steps 8
```

With Docker and Make:

```bash
make train
```

For a tiny CPU smoke test:

```bash
make train-smoke
```

The smoke test only verifies that the training code path works. It is not expected to
produce a useful classifier.

## Evaluate

```bash
python scripts/evaluate.py
```

With Docker and Make:

```bash
make evaluate
```

For a small Docker evaluation:

```bash
make evaluate-smoke MAX_SAMPLES=5
```

Evaluation writes:

- `outputs/metrics.json`
- `outputs/predictions.json`
- `outputs/confusion_matrix.png`

For a quick smoke evaluation:

```bash
python scripts/evaluate.py --max-samples 25
```

## Predict

Classify raw text:

```bash
python scripts/predict.py --text "Invoice number 1024. Total due 594.00. Payment terms net 30."
```

With Docker and Make:

```bash
make predict-text TEXT="Invoice number 1024. Total due 594.00. Payment terms net 30."
```

Classify a file:

```bash
python scripts/predict.py --file path\to\document.pdf
```

With Docker and Make:

```bash
make predict-file FILE=path/to/document.pdf
```

Supported file types:

- Text-like files: `.txt`, `.md`, `.csv`, `.json`, `.xml`, `.html`
- PDFs with embedded text
- Images supported by Pillow, using Tesseract OCR

## Results

The implementation is ready to run, but fine-tuning was not executed in this repository
setup because it requires downloading the model and dataset plus GPU time.

| Model | Dataset | Training Method | Samples | Accuracy | Macro F1 |
|---|---|---|---:|---:|---:|
| Qwen3-1.7B | RVL-CDIP OCR text | LoRA | Not run | Not run | Not run |

After training and evaluation, replace the final columns with values from
`outputs/metrics.json`.

## CPU-Only Startup Path

If you do not have GPU Docker support, use this sequence to verify the project yourself:

```bash
docker compose build
docker compose run --rm app python scripts/prepare_data.py --max-samples-per-split 5
docker compose run --rm app python scripts/predict.py --text "Invoice number 1024. Total due 594.00. Payment terms net 30."
```

The prediction command downloads Qwen3 1.7B the first time it runs. On CPU it can take a
while, but it should produce a category and the raw model answer.

To test that the fine-tuning script starts, run:

```bash
docker compose run --rm app python scripts/train.py --max-train-samples 2 --max-validation-samples 2 --max-length 512 --max-input-chars 1500 --epochs 1 --save-steps 1 --logging-steps 1
```

Then evaluate the tiny adapter:

```bash
make evaluate-smoke MAX_SAMPLES=5
```

Meaningful metrics require a real training run on enough data. CPU-only training is mainly
for verifying the workflow, not for producing final results.

## Colab Notes

If you see this error in Colab:

```text
RuntimeError: Dataset scripts are no longer supported, but found rvl_cdip.py
```

pull the latest branch and run preparation without overriding `--dataset-name`:

```bash
git pull
python scripts/prepare_data.py --max-samples-per-split 500
```

The project now defaults to `chainyo/rvl-cdip`, which is loaded as Parquet instead of
through the old `rvl_cdip.py` script.

## Repository Structure

```text
Dockerfile
docker-compose.yml
Makefile
scripts/
  prepare_data.py  # downloads RVL-CDIP and writes OCR text JSONL
  train.py         # LoRA/QLoRA fine-tuning
  evaluate.py      # accuracy, macro F1, classification report, confusion matrix
  predict.py       # single-document inference
src/doc_classifier/
  extraction.py    # PDF/image/text extraction
  labels.py        # category taxonomy and aliases
  modeling.py      # model loading and generation
  prompts.py       # instruction prompt formatting
  io.py            # JSONL helpers
```
