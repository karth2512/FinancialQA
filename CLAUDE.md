# FinancialQA Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-01-09

## Active Technologies

- Python 3.9+ (requires 3.9 minimum for type hinting features and modern ML library support) (001-finder-rag-system)

## Project Structure

```text
src/
├── agents/              # Multi-agent implementations
├── data/                # Dataset handling
├── evaluation/          # Evaluation pipeline
├── retrieval/           # Retrieval implementations
├── rag/                 # RAG pipelines
├── config/              # Configuration management
└── utils/               # Shared utilities
tests/
├── integration/
├── unit/
└── fixtures/
data/                    # Data directory (gitignored)
experiments/configs/     # Experiment configurations
```

## Commands

```bash
make setup              # Create venv and install dependencies
make download-data      # Download FinDER dataset
make eval-baseline      # Run baseline RAG evaluation
make eval-multiagent    # Run multi-agent evaluation
make lint               # Run black, flake8, mypy
make test               # Run pytest
make clean              # Remove generated files
```

## Code Style

Python 3.9+ (requires 3.9 minimum for type hinting features and modern ML library support): Follow standard conventions

## Recent Changes

- 001-finder-rag-system: Added Python 3.9+ (requires 3.9 minimum for type hinting features and modern ML library support)

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
