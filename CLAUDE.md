# FinancialQA Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-01-11

**NOTE:** This codebase focuses on the core workflow: **config-driven advanced RAG experiments with Langfuse tracking**. The system supports multiple RAG methodologies (baseline, query expansion) with different retrieval strategies.

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
│   ├── baseline.py           # BaselineRAG pipeline (CORE)
│   └── query_expansion.py    # QueryExpansionRAG pipeline (NEW)
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

## Supported RAG Pipelines

### Baseline RAG
- **Pipeline Type**: `baseline`
- **Flow**: Query → Retrieve → Generate
- **Config**: Single LLM (generator)
- **Use Case**: Simple, fast, cost-effective RAG

### Query Expansion RAG
- **Pipeline Type**: `query_expansion`
- **Flow**: Query → Expand (M variants) → Retrieve (M+1 times) → Pool & Deduplicate → Generate
- **Config**: Dual LLM (expander + generator)
- **Use Case**: Improved recall through query diversity
- **Trade-offs**:
  - +10-25% retrieval recall
  - +50-100% latency
  - +100-200% cost

### Configuration Examples

**Baseline**:
```yaml
pipeline_type: "baseline"
llm_configs:
  generator:
    provider: "anthropic"
    model: "claude-3-5-haiku-20241022"
```

**Query Expansion**:
```yaml
pipeline_type: "query_expansion"
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

## Parallel Subagents

**IMPORTANT**: When performing tasks that can be parallelized, always use multiple subagents (Task tool) in parallel to maximize efficiency.

### When to Use Parallel Subagents
- Searching for multiple unrelated patterns or files
- Researching different parts of the codebase simultaneously
- Running independent operations that don't depend on each other's results
- Exploring multiple hypotheses or approaches at once

### How to Parallelize
- Send a single message with multiple Task tool calls when the tasks are independent
- Example: If searching for "authentication" AND "authorization" patterns, spawn two Explore agents in parallel rather than sequentially
- Example: If researching how module A and module B work independently, use two codebase-researcher agents in parallel

### Do NOT Parallelize When
- One task depends on the result of another
- Tasks need to be executed in a specific order
- The tasks are modifying the same files

## Recent Changes

### 2026-01-14 - Query Expansion RAG - COMPLETED

**NEW FEATURE**: Added query expansion RAG methodology for improved retrieval recall.

- ✅ **QueryExpansionRAG pipeline** - Dual LLM setup (expander + generator)
- ✅ **Multi-query retrieval** - Retrieve for original + M expanded queries
- ✅ **Passage pooling** - Deduplicate by ID, keep highest scores
- ✅ **Config support** - New pipeline_type: "query_expansion"
- ✅ **Example config** - [experiments/configs/query_expansion.yaml](experiments/configs/query_expansion.yaml)

**How it works**:
1. Expander LLM generates M query variants (default M=3)
2. BM25 retrieval for each variant (M+1 total retrievals)
3. Pool passages and deduplicate by passage ID
4. Re-rank by score (descending)
5. Generator LLM produces final answer

**Trade-offs**:
- Recall: +10-25% (more relevant passages found)
- Latency: +50-100% (expansion + multiple retrievals)
- Cost: +100-200% (expansion tokens + more retrieval)

### 2026-01-11 - Codebase Simplification - COMPLETED

**CRITICAL FIX**: Experiment runner now uses actual RAG pipeline instead of placeholder code.

### What Changed
- ✅ **Fixed experiment runner** to use real BaselineRAG with BM25 retrieval
- ✅ **Deleted ~900 lines of unused code** (~60% reduction):
  - `src/data_handler/preprocessor.py` (315 lines) - never imported
  - `src/langfuse_integration/analysis.py` (~400 lines) - not used by main workflow
  - `src/langfuse_integration/client.py` (~50 lines) - redundant
  - `src/evaluation/runner.py`, `reporter.py` - old evaluation system
- ✅ **Simplified data_handler/indexer.py** - removed ChromaDB indexing (BM25 only)
- ✅ **Cleaned up imports** - removed references to deleted modules from [evaluation/models.py](src/evaluation/models.py)
- ✅ **Simplified config validation** - only "baseline" pipeline_type is valid in [experiments.py](src/config/experiments.py)
- ✅ **Marked dense/hybrid retrieval** as not actively tested in [indexer.py](src/data_handler/indexer.py)

### What to Know
- **Baseline and Query Expansion RAG pipelines are supported** - additional advanced RAG methods can be added
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

## GraphRAG

GraphRAG uses OpenAI's `gpt-4o-mini` (cheapest model) for entity extraction and community detection.

### Quick Start

```bash
# 1. Set API key in .env
cp .env.example .env
# Edit .env and add OPENAI_API_KEY

# 2. Initialize GraphRAG (first time only)
make graphrag-init

# 3. Configure GraphRAG settings
make graphrag-configure

# 4. Prepare input data (copy pre-chunked FinDER text files)
make graphrag-prepare-data

# 5. Run indexing (30-90 min)
make graphrag-index

# 6. Inspect the generated index
make graphrag-inspect

# 7. Run experiments
make graphrag-test        # Quick test with 10 items
make graphrag-baseline    # Full baseline experiment
make graphrag-global      # Global search experiment
make graphrag-qe          # Query expansion + GraphRAG
```

### Architecture

- **GraphRAG CLI** uses OpenAI API directly (`gpt-4o-mini`)
- **Embeddings** use local sentence-transformers (no API calls)
- **Data** is already pre-chunked in `data/finder_text/*.txt` (no additional chunking needed)

### Key Files

- GraphRAG workspace: `./ragtest/`
- Index outputs: `./ragtest/output/*/artifacts/*.parquet`
- Configuration scripts:
  - [scripts/configure_graphrag.py](scripts/configure_graphrag.py) - Auto-configure GraphRAG settings
  - [scripts/prepare_graphrag_input.py](scripts/prepare_graphrag_input.py) - Copy pre-chunked text files
  - [scripts/inspect_graphrag_index.py](scripts/inspect_graphrag_index.py) - Inspect parquet outputs
  - [scripts/graphrag_index.py](scripts/graphrag_index.py) - CLI wrapper for indexing

### Cost & Performance

**Indexing (one-time):**
- Duration: 30-90 minutes
- Cost: ~$1-2 USD (gpt-4o-mini is cheapest)
- Input: ~3,000 pre-chunked passages

**Querying (per experiment):**
- Local search: 1-3s latency
- Global search: 3-5s latency

### Troubleshooting

**GraphRAG workspace not initialized:**
```bash
# Initialize and configure
make graphrag-init
make graphrag-configure
```

**Input data missing:**
```bash
# Verify pre-chunked data exists
dir data\finder_text\*.txt

# Copy to GraphRAG input directory
make graphrag-prepare-data
```

**OPENAI_API_KEY not set:**
```bash
# Add to .env file
echo "OPENAI_API_KEY=sk-..." >> .env
```

## Git Workflow

**IMPORTANT**: Never create git commits automatically. All git operations (adding, commits, pushes, pull requests) will be handled manually by the user.
