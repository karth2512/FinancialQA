<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# FinancialQA Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-01-11

**NOTE:** This codebase has been simplified to focus on the core workflow: **config-driven Langchain RAG experiments with Langfuse tracking**. Multi-agent scaffolding and unused evaluation systems have been removed.

## Active Technologies

- Python 3.9+ (requires 3.9 minimum for type hinting features and modern ML library support)
- Langfuse SDK (datasets and experiments for evaluation tracking)
- BM25 retrieval (primary retrieval strategy)
- OpenAI/Anthropic LLMs for generation

## Project Structure

```text
src/
├── langfuse_integration/  # Langfuse experiment tracking (CORE)
│   ├── experiment_runner.py  # Main experiment orchestration
│   ├── evaluators.py         # Item & run-level evaluators
│   ├── models.py             # Pydantic models for results
│   └── config.py             # Langfuse config helpers
├── rag/
│   └── baseline.py           # BaselineRAG pipeline (CORE)
├── retrieval/
│   ├── base.py               # Retriever interface
│   ├── bm25.py               # BM25 retrieval (ACTIVELY USED)
│   ├── dense.py              # Dense retrieval (NOT actively tested)
│   └── hybrid.py             # Hybrid retrieval (NOT actively tested)
├── data_handler/
│   ├── loader.py             # FinDER dataset loader
│   └── models.py             # Data models (Query, EvidencePassage, etc.)
├── evaluation/
│   ├── metrics.py            # Evaluation metrics
│   └── evaluators.py         # Evaluator implementations
├── config/
│   └── experiments.py        # Pydantic config schemas
└── utils/
    ├── llm_client.py         # LLM client wrappers
    └── cache.py              # Caching utilities

scripts/
├── run_experiment.py         # Main entry point for experiments
├── upload_dataset.py         # Upload dataset to Langfuse
└── compare_experiments.py    # Compare experiment results

experiments/configs/          # Experiment YAML configs
data/                         # Data directory (gitignored)
```

## Detailed architecture design
Refer [ARCHITECTURE.md](ARCHITECTURE.md)ARCHITECTURE.md

## Commands

```bash
# Setup
make setup              # Create venv and install dependencies
make download-data      # Download FinDER dataset

# Run experiments
python scripts/run_experiment.py experiments/configs/langfuse_baseline.yaml
python scripts/run_experiment.py experiments/configs/baseline.yaml --max-items 10  # Quick test

# Upload dataset to Langfuse
python scripts/upload_dataset.py --name financial_qa_benchmark_v1 --filter all

# Linting and tests
make lint               # Run black, flake8, mypy
make test               # Run pytest
make clean              # Remove generated files
```

## Python Environment

**CRITICAL**: Always use the virtual environment when running Python commands.

- On Windows: Use `venv\Scripts\python` or activate with `venv\Scripts\activate`
- On Unix/MacOS: Use `venv/bin/python` or activate with `source venv/bin/activate`

Never use system Python or assume Python is available globally. All commands in this project must run within the venv to ensure correct dependencies.

## Code Style

Python 3.9+ (requires 3.9 minimum for type hinting features and modern ML library support): Follow standard conventions

## Recent Changes (2026-01-11 - Codebase Simplification - COMPLETED)

**CRITICAL FIX**: Experiment runner now uses actual RAG pipeline instead of placeholder code.

### What Changed
- ✅ **Fixed experiment runner** to use real BaselineRAG with BM25 retrieval
- ✅ **Deleted ~900 lines of unused code** (~60% reduction):
  - `src/data_handler/preprocessor.py` (315 lines) - never imported
  - `src/agents/base.py` (146 lines) - multi-agent scaffolding without implementations
  - `src/langfuse_integration/analysis.py` (~400 lines) - not used by main workflow
  - `src/langfuse_integration/client.py` (~50 lines) - redundant
  - `src/evaluation/runner.py`, `reporter.py` - old evaluation system
- ✅ **Simplified data_handler/indexer.py** - removed ChromaDB indexing (BM25 only)
- ✅ **Cleaned up imports** - removed references to deleted modules from [evaluation/models.py](src/evaluation/models.py)
- ✅ **Simplified config validation** - only "baseline" pipeline_type is valid in [experiments.py](src/config/experiments.py)
- ✅ **Marked dense/hybrid retrieval** as not actively tested in [indexer.py](src/data_handler/indexer.py)

### What to Know
- **Only baseline RAG pipeline is supported** - multi-agent removed, will be reimplemented when needed
- **BM25 is the primary retrieval strategy** - dense/hybrid exist but aren't actively tested
- **All experiments must use Langfuse** - old evaluation system removed
- **One config file per experiment** - experiment-specific settings in YAML
- **System defaults in [langfuse.yaml](src/config/langfuse.yaml)** - retry policies, batch sizes, etc.

### Previous Changes
- 001-finder-rag-system: Added Python 3.9+ (requires 3.9 minimum for type hinting features and modern ML library support)
- 001-langfuse-experiments: Added Langfuse SDK integration for dataset management and experiment tracking

## Langfuse Integration

The Langfuse integration provides comprehensive experiment tracking and evaluation for RAG experiments.

### Key Features
- **Dataset Management**: Upload and version FinDER datasets in Langfuse
- **Experiment Execution**: Run experiments with automatic tracing
- **Evaluation**: Item-level and run-level metrics (token F1, semantic similarity, retrieval quality)
- **Comparison**: Compare experiments and detect regressions

### Usage Examples

```bash
# Upload dataset to Langfuse
python scripts/upload_dataset.py --name financial_qa_benchmark_v1 --filter all

# Run experiment
python scripts/run_experiment.py experiments/configs/langfuse_baseline.yaml

# Compare experiments
python scripts/compare_experiments.py exp-001 exp-002 exp-003
```

### Configuration
Set Langfuse credentials in `.env`:
```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```
## Git Workflow

**IMPORTANT**: Never create git commits automatically. All git operations (adding, commits, pushes, pull requests) will be handled manually by the user.
