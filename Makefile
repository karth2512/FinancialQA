VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# For Windows compatibility
ifeq ($(OS),Windows_NT)
	PYTHON := $(VENV)/Scripts/python
	PIP := $(VENV)/Scripts/pip
endif

.PHONY: setup install download-data validate eval eval-baseline eval-multiagent eval-dev lint test clean help

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
	@echo "  make lint            - Run code quality checks (black, flake8, mypy)"
	@echo "  make test            - Run unit and integration tests"
	@echo "  make clean           - Remove generated files and caches"

setup: $(VENV)
	$(PIP) install -r requirements.txt
	@echo "Setup complete! Activate with: source venv/bin/activate (or venv\\Scripts\\activate on Windows)"

$(VENV):
	python3 -m venv $(VENV)

install:
	$(PIP) install -r requirements.txt

validate:
	$(PYTHON) scripts/validate_setup.py

download-data:
	$(PYTHON) -m src.data.loader --download

eval:
	$(PYTHON) -m src.evaluation.runner --config experiments/configs/latest.yaml

eval-baseline:
	$(PYTHON) -m src.evaluation.runner --config experiments/configs/baseline.yaml

eval-multiagent:
	$(PYTHON) -m src.evaluation.runner --config experiments/configs/multiagent.yaml

eval-dev:
	$(PYTHON) -m src.evaluation.runner --config experiments/configs/dev.yaml

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
