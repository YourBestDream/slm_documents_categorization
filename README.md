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

For training and evaluation, prefer balanced samples:

```bash
python scripts/prepare_data.py --samples-per-label 25
```

This writes up to 25 OCR-valid examples for each of the 16 labels in each split. The script
prints the label distribution after writing every split. If the test set contains only one
label, the evaluation metrics are not meaningful.

Progress is printed every 25 written records by default. Change it with:

```bash
python scripts/prepare_data.py --samples-per-label 150 --progress-every 10
```

With Docker and Make:

```bash
make prepare
```

Without Make:

```bash
docker compose run --rm app python scripts/prepare_data.py --max-samples-per-split 25
```

Balanced Docker sample:

```bash
docker compose run --rm app python scripts/prepare_data.py --samples-per-label 25
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

Evaluation uses strict label scoring by default, so predictions are always selected from
the configured label list. Use `--mode generate` only when comparing with the original
free-text generation behavior.

For a quick smoke evaluation:

```bash
python scripts/evaluate.py --max-samples 25
```

Progress is printed every 25 evaluated records by default. Change it with:

```bash
python scripts/evaluate.py --progress-every 10
```

## Predict

Classify raw text:

```bash
python scripts/predict.py --text "Invoice number 1024. Total due 594.00. Payment terms net 30."
```

Prediction uses strict label scoring by default, so `category` is always one of the
configured labels. To inspect scores:

```bash
python scripts/predict.py --text "Invoice number 1024. Total due 594.00." --show-scores
```

The legacy free-text generation mode is still available for debugging:

```bash
python scripts/predict.py --text "Invoice number 1024. Total due 594.00." --mode generate
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

Fine-tuning was run on Kaggle with `Qwen/Qwen3-1.7B`, 4-bit loading, and LoRA adapters.
The experiment used OCR text from balanced RVL-CDIP samples:

- Train: 1,440 documents, 90 per class
- Validation: 640 documents, 40 per class
- Test: 640 documents, 40 per class
- Epochs: 1
- Training steps: 180
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05

| Model | Dataset | Training Method | Train Samples | Test Samples | Accuracy | Macro F1 |
|---|---|---|---:|---:|---:|---:|
| Qwen3-1.7B | RVL-CDIP OCR text | QLoRA | 1,440 | 640 | 0.7063 | 0.7075 |

Per-class F1 on the balanced test set:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| resume | 0.975 | 0.975 | 0.975 |
| email | 0.875 | 0.875 | 0.875 |
| specification | 0.854 | 0.875 | 0.864 |
| questionnaire | 0.892 | 0.825 | 0.857 |
| scientific publication | 0.889 | 0.800 | 0.842 |
| news article | 0.750 | 0.900 | 0.818 |
| letter | 0.718 | 0.700 | 0.709 |
| memo | 0.718 | 0.700 | 0.709 |
| form | 0.733 | 0.550 | 0.629 |
| invoice | 0.649 | 0.600 | 0.623 |
| presentation | 0.710 | 0.550 | 0.620 |
| scientific report | 0.800 | 0.500 | 0.615 |
| file folder | 0.435 | 0.925 | 0.592 |
| advertisement | 0.564 | 0.550 | 0.557 |
| budget | 0.613 | 0.475 | 0.535 |
| handwritten | 0.500 | 0.500 | 0.500 |

The strongest classes were `resume`, `email`, `specification`, `questionnaire`, and
`scientific publication`. The weakest classes were `handwritten`, `budget`,
`advertisement`, and `file folder`. `file folder` had high recall but low precision,
which means the model over-predicted that category.

Evaluation artifacts from the run:

- `outputs/metrics.json`
- `outputs/predictions.json`
- `outputs/confusion_matrix.png`

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
