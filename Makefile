VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: setup install download-data validate eval eval-baseline eval-multiagent eval-dev run-experiment lint test clean help

help:
	@echo "FinDER Multi-Agent RAG System - Available targets:"
	@echo "  make setup           - Create virtual environment and install dependencies"
	@echo "  make install         - Install dependencies (assumes venv exists)"
	@echo "  make validate        - Validate setup and configuration"
	@echo "  make download-data   - Download FinDER dataset from HuggingFace"
	@echo "  make eval            - Run latest evaluation configuration"
	@echo "  make eval-baseline   - Run baseline RAG evaluation"
	@echo "  make eval-multiagent - Run multi-agent RAG evaluation"
	@echo "  make eval-dev        - Run development subset evaluation (fast iteration)"
	@echo "  make run-experiment  - Run Langfuse experiment (CONFIG=path/to/config.yaml)"
	@echo "  make lint            - Run code quality checks (black, flake8, mypy)"
	@echo "  make test            - Run unit and integration tests"
	@echo "  make clean           - Remove generated files and caches"

setup: $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Setup complete! Activate with: source venv/bin/activate"

$(VENV):
	python3 -m venv $(VENV)
	$(VENV)/bin/python -m ensurepip --upgrade

install:
	$(PIP) install -r requirements.txt

validate:
	$(PYTHON) scripts/validate_setup.py

download-data:
	$(PYTHON) -m src.data_handler.loader --download

lf-run:
	$(PYTHON) scripts/upload_dataset.py --name financial_qa_benchmark_v1 --filter all --yes --max-items 10

eval:
	$(PYTHON) -m src.evaluation.runner --config experiments/configs/latest.yaml

eval-baseline:
	$(PYTHON) -m src.evaluation.runner --config experiments/configs/baseline.yaml

eval-multiagent:
	$(PYTHON) -m src.evaluation.runner --config experiments/configs/multiagent.yaml

eval-dev:
	$(PYTHON) -m src.evaluation.runner --config experiments/configs/dev.yaml

CONFIG ?= experiments/configs/langfuse_baseline.yaml

run-experiment:
	$(PYTHON) scripts/run_experiment.py $(CONFIG)
te:
	$(PYTHON) test_langfuse_experiment.py

lint:
	$(PYTHON) -m black src tests --check
	$(PYTHON) -m flake8 src tests
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	rm -rf data/embeddings data/cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type f -name '*.log' -delete
