VENV := venv
ifeq ($(OS),Windows_NT)
    PYTHON := $(VENV)\Scripts\python
    PIP := $(VENV)\Scripts\pip
else
    PYTHON := $(VENV)/bin/python
    PIP := $(VENV)/bin/pip
endif

.PHONY: setup install download-data validate eval eval-baseline eval-dev run-experiment run-qe-test graphrag-init graphrag-configure graphrag-prepare-data graphrag-index graphrag-index-python graphrag-inspect graphrag-test graphrag-baseline graphrag-global graphrag-qe lint test clean help

help:
	@echo "FinDER Advanced RAG System - Available targets:"
	@echo "  make setup           - Create virtual environment and install dependencies"
	@echo "  make install         - Install dependencies (assumes venv exists)"
	@echo "  make validate        - Validate setup and configuration"
	@echo "  make download-data   - Download FinDER dataset from HuggingFace"
	@echo "  make eval            - Run latest evaluation configuration"
	@echo "  make eval-baseline   - Run baseline RAG evaluation"
	@echo "  make eval-dev        - Run development subset evaluation (fast iteration)"
	@echo "  make run-experiment  - Run Langfuse experiment (CONFIG=path/to/config.yaml)"
	@echo "  make run-qe-test     - Run query expansion experiment with 10 items (quick test)"
	@echo ""
	@echo "GraphRAG Targets:"
	@echo "  make graphrag-index        - Build GraphRAG index via CLI (uses OpenAI API)"
	@echo "  make graphrag-index-python - Build GraphRAG index using Python API"
	@echo "  make graphrag-init         - Initialize GraphRAG workspace (first-time setup)"
	@echo "  make graphrag-configure    - Auto-configure GraphRAG settings"
	@echo "  make graphrag-prepare-data - Copy pre-chunked FinDER text to input dir"
	@echo "  make graphrag-inspect      - Inspect generated parquet files"
	@echo "  make graphrag-test         - Quick test GraphRAG with 10 items"
	@echo "  make graphrag-baseline     - Run GraphRAG baseline (local search)"
	@echo "  make graphrag-global       - Run GraphRAG global search"
	@echo "  make graphrag-qe           - Run query expansion + GraphRAG"
	@echo ""
	@echo "Other Targets:"
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

CONFIG ?= experiments/configs/langfuse_baseline.yaml

run-experiment:
	$(PYTHON) scripts/run_experiment.py $(CONFIG)

run-qe-test:
	$(PYTHON) scripts/run_experiment.py experiments/configs/query_expansion.yaml --max-items 3

# GraphRAG Setup
graphrag-init:
	@echo "Initializing GraphRAG workspace"
	graphrag init --root ./ragtest
	@echo GraphRAG initialized! Next steps:
	@echo 1. Run 'make graphrag-configure' to auto-configure settings
	@echo 2. Run 'make graphrag-prepare-data' to copy FinDER text files
	@echo 3. Run 'make graphrag-index' to build the index

graphrag-configure:
	@echo "Auto-configuring GraphRAG settings for OpenAI (gpt-4o-mini) + local embeddings"
	$(PYTHON) scripts/configure_graphrag.py

graphrag-prepare-data:
	@echo "Copying pre-chunked FinDER text files to GraphRAG input directory"
	$(PYTHON) scripts/prepare_graphrag_input.py

graphrag-inspect:
	@echo "Inspecting GraphRAG index files"
	$(PYTHON) scripts/inspect_graphrag_index.py

# GraphRAG Indexing (CLI - uses OpenAI API directly)
graphrag-index:
	@echo "Building GraphRAG index via CLI (this will take 30-90 minutes)"
	@echo "Using OpenAI gpt-4o-mini (cheapest model)"
	@echo "Prerequisites:"
	@echo "  1. OPENAI_API_KEY set in .env"
	@echo "  2. GraphRAG initialized and configured (make graphrag-init + make graphrag-configure)"
	@echo "  3. Input data prepared (make graphrag-prepare-data)"
	@echo ""
	graphrag index --root ./ragtest --verbose

graphrag-test:
	@echo "Running quick GraphRAG test (10 items)"
	$(PYTHON) scripts/run_experiment.py experiments/configs/graphrag_baseline.yaml --max-items 5

graphrag-baseline:
	@echo "Running GraphRAG baseline experiment (local search)"
	$(PYTHON) scripts/run_experiment.py experiments/configs/graphrag_baseline.yaml

graphrag-global:
	@echo "Running GraphRAG global search experiment"
	$(PYTHON) scripts/run_experiment.py experiments/configs/graphrag_global.yaml

graphrag-qe:
	@echo "Running query expansion + GraphRAG experiment"
	$(PYTHON) scripts/run_experiment.py experiments/configs/graphrag_query_expansion.yaml

te:
	$(PYTHON) test_langfuse_experiment.py

arr-to-text:
	$(PYTHON) scripts/convert_arrow_to_text.py 

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
