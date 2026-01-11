# Experiment Configuration Guide

This directory contains YAML configuration files for running RAG experiments on the FinDER dataset.

## Overview

Experiment configurations define all parameters needed to run an evaluation:
- **Retrieval strategy**: BM25, dense embeddings, or hybrid
- **LLM configuration**: Model, provider, temperature, token limits
- **Evaluation settings**: Dataset, metrics, Langfuse integration
- **Execution parameters**: Concurrency, tracing, evaluation thresholds

All configurations are validated against Pydantic schemas in `src/config/experiments.py`.

## Configuration Files

### Active Configurations

- **baseline.yaml** - Standard BM25 + Claude 3.5 Haiku baseline
  - Full dataset evaluation
  - BM25 retrieval (k1=1.5, b=0.75, top_k=5)
  - 256 max tokens
  - Use: `make eval-baseline`

- **dev.yaml** - Fast iteration config for development
  - 10-query subset (first 10 queries)
  - BM25 retrieval (k1=1.5, b=0.75, top_k=3)
  - 128 max tokens
  - Use: `make eval-dev`

- **langfuse_baseline.yaml** - Baseline with full Langfuse integration
  - Loads dataset from Langfuse: "financial_qa_benchmark_v1"
  - Full tracing and evaluation
  - 4 item-level evaluators: token_f1, semantic_similarity, retrieval_precision, retrieval_recall
  - 3 run-level evaluators: average_accuracy, aggregate_retrieval_metrics, pass_rate
  - Use: `make run-experiment CONFIG=experiments/configs/langfuse_baseline.yaml`

### Missing Configurations (Referenced but Don't Exist)

- **latest.yaml** - Referenced in Makefile line 48, needs creation
- **multiagent.yaml** - Referenced in Makefile line 54, multiagent not implemented yet

## Configuration Schema

### Base Schema: ExperimentConfig

All experiment configs must include:

```yaml
name: "Descriptive Experiment Name"
description: "What this experiment tests and why"
run_id: "unique-identifier-v1.0.0"
pipeline_type: "baseline"  # or "multiagent", "specialized"

retrieval_config:
  strategy: "bm25"  # or "dense", "hybrid"
  top_k: 5
  # BM25-specific (optional, defaults shown):
  k1: 1.5
  b: 0.75
  # Dense/Hybrid-specific (optional):
  embedding_model: "all-MiniLM-L6-v2"
  reranking: false
  bm25_weight: 0.5  # for hybrid only
  dense_weight: 0.5  # for hybrid only

llm_configs:
  generator:  # for baseline, use agent names for multiagent
    provider: "anthropic"  # or "openai", "local"
    model: "claude-3-5-haiku-20241022"
    temperature: 0.0
    max_tokens: 256
    api_key_env_var: "ANTHROPIC_API_KEY"

# Optional: additional hyperparameters
hyperparameters:
  subset_size: 100  # limit dataset for testing
  sample_strategy: "first"  # or "random"
```

### Extended Schema: LangfuseExperimentConfig

For Langfuse-integrated experiments, add:

```yaml
# Dataset Configuration
langfuse_dataset_name: "financial_qa_benchmark_v1"
use_local_data: false  # true to use local FinDER data instead
local_data_path: null  # path if use_local_data=true

# Tracing Configuration
flush_interval: 5.0  # seconds between automatic flushes
flush_batch_size: 100  # events before flush
enable_tracing: true

# Concurrency
max_concurrency: 1  # 1=sequential, higher for parallel (respect rate limits)

# Evaluators
enable_item_evaluators: true
enable_run_evaluators: true

item_evaluator_names:
  - "token_f1"
  - "semantic_similarity"
  - "retrieval_precision"
  - "retrieval_recall"

run_evaluator_names:
  - "average_accuracy"
  - "aggregate_retrieval_metrics"
  - "pass_rate"

# Evaluation Thresholds (for pass_rate evaluator)
evaluation_thresholds:
  token_f1: 0.5
  retrieval_precision: 0.4

# Metadata and Tags
langfuse_tags:
  - "experiment:baseline"
  - "model:claude-3.5-haiku"
  - "retrieval:bm25"

langfuse_metadata:
  team: "research"
  environment: "development"
  version: "1.0"

propagate_query_metadata: true
```

## Schema Validation

All configs are validated using Pydantic models:
- **LLMConfig**: Validates provider in {openai, anthropic, local}
- **RetrievalConfig**: Validates strategy in {bm25, dense, hybrid}
- **ExperimentConfig**: Base experiment validation
- **LangfuseExperimentConfig**: Extended Langfuse validation

Invalid configurations will fail fast with clear error messages.

## Usage Patterns

### Running Standard Evaluations

```bash
# Development iteration (10 queries)
make eval-dev

# Full baseline evaluation
make eval-baseline

# Custom config
python -m src.evaluation.runner --config experiments/configs/your_config.yaml
```

### Running Langfuse Experiments

```bash
# Default Langfuse baseline
make run-experiment

# Custom Langfuse config
make run-experiment CONFIG=experiments/configs/your_langfuse_config.yaml

# With CLI overrides
python scripts/run_experiment.py experiments/configs/langfuse_baseline.yaml --concurrency 5
python scripts/run_experiment.py experiments/configs/langfuse_baseline.yaml --dry-run
python scripts/run_experiment.py experiments/configs/langfuse_baseline.yaml --disable-evaluators
```

### CLI Overrides

The `run_experiment.py` script supports command-line overrides:
- `--concurrency N` - Override max_concurrency
- `--max-items N` - Limit dataset size
- `--disable-evaluators` - Skip all evaluators
- `--dry-run` - Validate config without running
- `--output PATH` - Save results JSON

## Configuration Conventions

### Naming Conventions

- **Experiment Names**: Descriptive, human-readable (e.g., "Baseline RAG Evaluation")
- **Run IDs**: Versioned or dated identifiers (e.g., "v1.0.0-baseline", "exp-2026-01-10-baseline")
- **File Names**: lowercase with underscores (e.g., `langfuse_baseline.yaml`)

### Best Practices

1. **Never modify configs after experiments run** - Create new versions instead
2. **Use semantic versioning** - Increment run_id for each variation
3. **Document in description field** - Explain what's being tested
4. **Tag appropriately** - Use langfuse_tags for filtering and organization
5. **Start with small subsets** - Use hyperparameters.subset_size for testing
6. **Version control all configs** - Commit configs with experiment results

### Common Patterns

**Fast Development Iteration**:
```yaml
hyperparameters:
  subset_size: 10
  sample_strategy: "first"
max_concurrency: 1
enable_item_evaluators: false  # skip for speed
```

**Production Evaluation**:
```yaml
# Full dataset, all evaluators
enable_item_evaluators: true
enable_run_evaluators: true
max_concurrency: 1  # respect rate limits
```

**A/B Testing**:
```yaml
# Version A: baseline_v1.yaml
run_id: "exp-001-baseline-v1"
retrieval_config:
  top_k: 5

# Version B: baseline_v2.yaml
run_id: "exp-001-baseline-v2"
retrieval_config:
  top_k: 10
```

## Known Issues and Gotchas

### Current Bugs

1. **Duplicate field in langfuse_baseline.yaml**: `langfuse_dataset_name` appears twice (lines 26, 78) - delete one
2. **Missing latest.yaml**: Makefile references this but it doesn't exist

### Unused Parameters

The following are defined in schemas but never used in practice:
- `reranking` - Always false, no implementation exists
- `agent_architecture` - Multiagent not implemented
- `orchestration_mode` - Multiagent not implemented
- `use_local_data` - Always false, local loading stubbed

### Hard-coded Values

Some values are hard-coded in source and cannot be configured:
- Prompt templates (`src/rag/baseline.py` lines 101-130)
- BM25 tokenization strategy (whitespace only)
- Dense embedding batch size (32)
- ChromaDB persistence path (`./data/embeddings`)

## Relationship with src/config/

The `src/config/` directory contains:
- **experiments.py** - Pydantic schema definitions and validation
- **langfuse.yaml** - System-wide Langfuse SDK defaults

**How they interact**:
- `langfuse.yaml` provides defaults for flush intervals, retry logic, etc.
- Experiment configs can override these per-experiment
- Schemas in `experiments.py` validate both standard and Langfuse configs

**Settings that overlap**:
- `flush_interval`, `flush_batch_size` - Default in langfuse.yaml, overridable in experiment configs
- `enable_tracing` - Can be disabled system-wide or per-experiment
- Evaluator defaults - System defaults in langfuse.yaml, specific lists in experiment configs

## Examples and Templates

### Minimal BM25 Baseline

```yaml
name: "Simple BM25 Test"
description: "Minimal config for BM25 retrieval"
run_id: "test-001"
pipeline_type: "baseline"

retrieval_config:
  strategy: "bm25"
  top_k: 5

llm_configs:
  generator:
    provider: "anthropic"
    model: "claude-3-5-haiku-20241022"
    temperature: 0.0
    max_tokens: 256
    api_key_env_var: "ANTHROPIC_API_KEY"

hyperparameters: {}
```

### Dense Retrieval (Future)

```yaml
name: "Dense Semantic Retrieval"
description: "Semantic search with sentence transformers"
run_id: "dense-001"
pipeline_type: "baseline"

retrieval_config:
  strategy: "dense"
  top_k: 5
  embedding_model: "all-MiniLM-L6-v2"
  reranking: false

llm_configs:
  generator:
    provider: "anthropic"
    model: "claude-3-5-haiku-20241022"
    temperature: 0.0
    max_tokens: 256
    api_key_env_var: "ANTHROPIC_API_KEY"

hyperparameters: {}
```

### Hybrid Retrieval (Future)

```yaml
name: "Hybrid BM25 + Dense"
description: "Combine keyword and semantic search"
run_id: "hybrid-001"
pipeline_type: "baseline"

retrieval_config:
  strategy: "hybrid"
  top_k: 5
  embedding_model: "all-MiniLM-L6-v2"
  bm25_weight: 0.6
  dense_weight: 0.4
  fusion_method: "reciprocal_rank"  # or "score_fusion"
  k1: 1.5  # BM25 params
  b: 0.75

llm_configs:
  generator:
    provider: "anthropic"
    model: "claude-3-5-haiku-20241022"
    temperature: 0.0
    max_tokens: 256
    api_key_env_var: "ANTHROPIC_API_KEY"

hyperparameters: {}
```

## Validation and Debugging

### Validate Config Before Running

```bash
# Dry run mode validates without executing
python scripts/run_experiment.py experiments/configs/your_config.yaml --dry-run
```

### Common Validation Errors

**Invalid provider**:
```
ValueError: provider must be one of {'openai', 'anthropic', 'local'}
```
Fix: Check llm_configs.generator.provider

**Invalid strategy**:
```
ValueError: strategy must be one of {'bm25', 'dense', 'hybrid'}
```
Fix: Check retrieval_config.strategy

**Missing required fields**:
```
ValidationError: field required
```
Fix: Add all required fields (name, description, run_id, etc.)

**Unknown evaluator**:
```
ValueError: Unknown item evaluator: xyz. Valid options: {...}
```
Fix: Use valid evaluator names from schema

### Debugging Tips

1. **Start with dev.yaml** - Known working config with small subset
2. **Use --dry-run** - Validate before executing
3. **Check credentials** - Ensure .env has required API keys
4. **Review logs** - Experiment runner logs config loading and validation
5. **Compare with examples** - Use baseline.yaml as reference

## Future Enhancements

Planned improvements to the config system:
- [ ] Config presets (e.g., `retrieval_preset: "bm25_default"`)
- [ ] Prompt template configuration
- [ ] Config migration utilities
- [ ] Auto-generated config documentation from schemas
- [ ] Config validation warnings for unused parameters
- [ ] Examples directory with templates for all strategies

## References

- **Schema definitions**: `src/config/experiments.py`
- **Langfuse defaults**: `src/config/langfuse.yaml`
- **Standard runner**: `src/evaluation/runner.py`
- **Langfuse runner**: `scripts/run_experiment.py`
- **RAG pipeline**: `src/rag/baseline.py`
- **Retrieval implementations**: `src/retrieval/`
