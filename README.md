# FinDER Multi-Agent Financial RAG System

A multi-agent Retrieval-Augmented Generation (RAG) system for answering financial questions using the FinDER dataset. This project implements baseline and multi-agent approaches with comprehensive evaluation and observability through Langfuse.

## Overview

This system evaluates different RAG architectures on the FinDER dataset (5,703 financial query-evidence-answer triplets), comparing:
- **Baseline**: Single-agent RAG (query → retrieve → generate)
- **Query Expansion**: Multi-query RAG (expand → retrieve multiple → pool → generate)

All development follows evaluation-first principles with metrics logged to Langfuse for performance tracking and regression detection.

## Quick Start

### Prerequisites

- Python 3.9+
- 8GB+ RAM recommended
- Optional: GPU for faster embedding generation

### Setup

```bash
# Create virtual environment and install dependencies
make setup

# Download FinDER dataset
make download-data

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys (OpenAI, Anthropic, Langfuse)
```

### Run Baseline Evaluation

```bash
# Run baseline single-agent RAG evaluation
python scripts/run_experiment.py experiments/configs/langfuse_baseline.yaml --max-items 10

# Uses BM25 retrieval + Claude Haiku generation
```

### Run Query Expansion Evaluation

```bash
# Run query expansion RAG with dual LLMs
python scripts/run_experiment.py experiments/configs/query_expansion.yaml --max-items 10

# Uses:
# - GPT-3.5 for query expansion (3 variants)
# - BM25 retrieval (4 queries × 5 passages = max 20)
# - Claude Haiku for answer generation
# - Passage pooling and deduplication
```

## Project Structure

```
src/
├── rag/                 # RAG pipelines
│   ├── baseline.py      # Single-agent RAG
│   └── query_expansion.py  # Multi-query expansion RAG
├── retrieval/           # Retrieval implementations
│   ├── bm25.py          # BM25 keyword search
│   ├── dense.py         # Dense embedding search
│   └── hybrid.py        # Hybrid retrieval
├── langfuse_integration/  # Experiment tracking
│   ├── experiment_runner.py
│   ├── evaluators.py
│   └── models.py
├── data_handler/        # Dataset handling
│   ├── loader.py        # FinDER dataset loader
│   └── models.py        # Data models
├── config/              # Configuration management
│   └── experiments.py   # Experiment config loader
└── utils/               # Shared utilities
    ├── llm_client.py    # LLM API wrapper
    └── cache.py         # Caching utilities

scripts/                 # Entry points
experiments/configs/     # Experiment configurations
data/                    # Data directory (gitignored)
```

## Available Commands

```bash
make setup              # Create venv, install dependencies
make install            # Install dependencies (venv exists)
make download-data      # Download FinDER dataset
make eval               # Run latest evaluation config
make eval-baseline      # Run baseline evaluation
make eval-multiagent    # Run multi-agent evaluation
make eval-dev           # Run dev subset (fast iteration)
make lint               # Run code quality checks
make test               # Run unit/integration tests
make clean              # Remove generated files

# Data conversion
python scripts/convert_arrow_to_text.py  # Convert Arrow dataset to individual text files
```

## Configuration

Experiment configurations are stored in `experiments/configs/` as YAML files.

### Baseline RAG Example

```yaml
name: "Baseline RAG Evaluation"
pipeline_type: "baseline"
retrieval_config:
  strategy: "bm25"
  top_k: 5
llm_configs:
  generator:
    provider: "anthropic"
    model: "claude-3-5-haiku-20241022"
```

### Query Expansion RAG Example

```yaml
name: "Query Expansion RAG"
pipeline_type: "query_expansion"
retrieval_config:
  strategy: "bm25"
  top_k: 5
llm_configs:
  expander:
    provider: "openai"
    model: "gpt-3.5-turbo"
    temperature: 0.7
  generator:
    provider: "anthropic"
    model: "claude-3-5-haiku-20241022"
    temperature: 0.0
hyperparameters:
  num_expanded_queries: 3
```

## Evaluation Metrics

The system tracks comprehensive metrics for each evaluation run:

**Answer Correctness**:
- Exact match rate
- Token-level F1 score
- Semantic similarity (embedding cosine)
- LLM-as-judge scoring (optional)

**Retrieval Quality**:
- Precision (% retrieved that are relevant)
- Recall (% relevant that are retrieved)
- F1 score
- Mean Reciprocal Rank (MRR)

**Operational Metrics**:
- End-to-end latency per query
- LLM token usage
- Cost per query
- Total evaluation cost

## Langfuse Integration

All evaluation runs are logged to Langfuse with:
- Unique `run_id` or version identifier
- Complete query traces (input, retrieval, generation, metrics)
- Agent decision logs (for multi-agent runs)
- Aggregate metrics and performance breakdowns

View results at: https://cloud.langfuse.com

## Development

For rapid iteration, use the development subset:

```bash
# Run on 500 queries (10% of dataset) for fast feedback
make eval-dev
```

### Code Quality

```bash
# Run linters and type checking
make lint

# Run tests
make test
```

## Documentation

- [Feature Specification](specs/001-finder-rag-system/spec.md)
- [Implementation Plan](specs/001-finder-rag-system/plan.md)
- [Data Model](specs/001-finder-rag-system/data-model.md)
- [API Contracts](specs/001-finder-rag-system/contracts/)
- [Quickstart Guide](specs/001-finder-rag-system/quickstart.md)

## Tech Stack

- **Python 3.9+**: Core language with type hints
- **ML/NLP**: transformers, sentence-transformers, torch
- **Retrieval**: rank_bm25, ChromaDB, FAISS
- **Evaluation**: HuggingFace evaluate, scikit-learn
- **Observability**: Langfuse (Python SDK)
- **LLM Integration**: OpenAI, Anthropic APIs
- **Configuration**: Pydantic, PyYAML

## License

This project is for research and educational purposes.

## Contributing

This is a research project. For questions or issues, please refer to the documentation in `specs/001-finder-rag-system/`.
