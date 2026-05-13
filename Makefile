COMPOSE ?= docker compose
SERVICE ?= app
MAX_SAMPLES ?= 25
TEXT ?= Invoice number 1024. Total due 594.00. Payment terms net 30.
FILE ?=

.PHONY: help docker-build shell prepare prepare-full train train-smoke evaluate evaluate-smoke predict-text predict-file smoke clean-artifacts

help:
	@echo "Available targets:"
	@echo "  make docker-build       Build the CPU Docker image with Tesseract"
	@echo "  make shell              Open a shell inside the container"
	@echo "  make prepare            Prepare a small OCR dataset sample"
	@echo "  make prepare-full       Prepare full RVL-CDIP splits"
	@echo "  make train              Fine-tune Qwen3 1.7B with LoRA on CPU"
	@echo "  make train-smoke        Run a tiny CPU training smoke test"
	@echo "  make evaluate           Evaluate the trained adapter"
	@echo "  make evaluate-smoke     Evaluate a small number of examples"
	@echo "  make predict-text       Classify TEXT='...'"
	@echo "  make predict-file       Classify FILE=path/to/document.pdf"
	@echo "  make smoke              Build image and run a text prediction smoke test"
	@echo "  make clean-artifacts    Remove generated data, model, and output files"

docker-build:
	$(COMPOSE) build

shell:
	$(COMPOSE) run --rm $(SERVICE) bash

prepare:
	$(COMPOSE) run --rm $(SERVICE) python scripts/prepare_data.py --max-samples-per-split $(MAX_SAMPLES)

prepare-full:
	$(COMPOSE) run --rm $(SERVICE) python scripts/prepare_data.py --max-samples-per-split 0

train:
	$(COMPOSE) run --rm $(SERVICE) python scripts/train.py

train-smoke:
	$(COMPOSE) run --rm $(SERVICE) python scripts/train.py --max-train-samples 2 --max-validation-samples 2 --max-length 512 --max-input-chars 1500 --epochs 1 --save-steps 1 --logging-steps 1

evaluate:
	$(COMPOSE) run --rm $(SERVICE) python scripts/evaluate.py

evaluate-smoke:
	$(COMPOSE) run --rm $(SERVICE) python scripts/evaluate.py --max-samples $(MAX_SAMPLES)

predict-text:
	$(COMPOSE) run --rm $(SERVICE) python scripts/predict.py --text "$(TEXT)"

predict-file:
	$(COMPOSE) run --rm $(SERVICE) python scripts/predict.py --file "$(FILE)"

smoke: docker-build predict-text

clean-artifacts:
	$(COMPOSE) run --rm $(SERVICE) sh -c 'find data/processed models outputs -type f ! -name .gitkeep -delete'
