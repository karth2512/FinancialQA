# Langfuse Integration Module

## Purpose

Central hub for Langfuse SDK integration, providing experiment execution, evaluation, dataset management, and configuration handling. This module orchestrates the entire RAG experiment workflow.

## Key Components

### 1. Experiment Runner ([experiment_runner.py](experiment_runner.py))

**Main Entry Point:** `run_langfuse_experiment(config, langfuse_client=None)`

Orchestrates the complete experiment workflow:
1. Initialize Langfuse client (or use provided)
2. Load dataset from Langfuse or local source
3. Create task function (RAG pipeline)
4. Register evaluators (item and run-level)
5. Execute experiment via Langfuse SDK
6. Transform and return results

**Task Function Creation:** `create_task_function(config)`
- Factory pattern that returns a task function based on `config.pipeline_type`
- Currently supports: `"baseline"` (only valid option)
- Task function signature: `(*, item, **kwargs) -> Dict[str, Any]` (Langfuse convention)

**Baseline Task:** `_create_baseline_task(config)`
- Loads corpus via `FinDERLoader`
- Creates retriever (BM25/dense/hybrid based on config)
- Indexes corpus once (closure captures index)
- Initializes LLM client
- Creates `BaselineRAG` pipeline
- Returns task function that executes RAG per item

### 2. Evaluators ([evaluators.py](evaluators.py))

**Evaluator Registration:** `register_evaluators(config)`

Returns two lists of evaluator functions:
- **Item-level evaluators** - Called once per query
- **Run-level evaluators** - Called once after all queries

#### Item-Level Evaluators

Signature: `(*, input, output, expected_output, metadata, **kwargs) -> Evaluation`

Available evaluators:
- `evaluate_token_f1()` - Token overlap F1 between generated and expected answer
- `evaluate_exact_match()` - Binary exact string match (0/1)
- `SemanticSimilarityEvaluator` - Cosine similarity using sentence-transformers
- `evaluate_retrieval_precision()` - Relevant retrieved / total retrieved
- `evaluate_retrieval_recall()` - Relevant found / total expected evidence
- `evaluate_retrieval_quality()` - F1 of precision and recall

#### Run-Level Evaluators

Signature: `(*, item_results, **kwargs) -> Evaluation`

Available evaluators:
- `compute_average_accuracy()` - Mean token F1 across all items
- `compute_aggregate_retrieval_metrics()` - Mean precision/recall/F1
- `PassRateEvaluator` - Percentage of items meeting threshold criteria

### 3. Data Models ([models.py](models.py))

#### Core Result Models

**`ExperimentRunOutput`** - Complete experiment result container
```python
ExperimentRunOutput(
    run_id: str,
    experiment_name: str,
    dataset_name: str,
    item_results: List[ItemExecutionResult],
    aggregate_scores: Dict[str, EvaluationScore],
    status: str,  # "completed", "partial", "failed"
    langfuse_url: str,
    execution_metadata: Dict
)
```

**`ItemExecutionResult`** - Per-query result
```python
ItemExecutionResult(
    item_id: str,
    input: Dict[str, Any],
    output: Dict[str, Any],
    expected_output: Optional[str],
    evaluations: List[EvaluationScore],
    trace_id: Optional[str],
    latency_seconds: float,
    error: Optional[str]
)
```

**`EvaluationScore`** - Individual metric score
```python
EvaluationScore(
    name: str,
    value: float,
    data_type: str,  # "NUMERIC", "CATEGORICAL", "BOOLEAN"
    string_value: Optional[str],
    comment: Optional[str]
)
```

#### Dataset Models

**`LangfuseDatasetConfig`** - Dataset upload configuration
```python
LangfuseDatasetConfig(
    dataset_name: str,
    dataset_description: str,
    filter_criteria: FilterCriteria,  # ALL, REASONING_REQUIRED, etc.
    metadata_options: MetadataOptions,
    upload_options: UploadOptions,
    version: Optional[str]
)
```

**`DatasetItemMapping`** - Transform FinDER Query to Langfuse item
```python
DatasetItemMapping.transform_query(query: Query) -> Dict[str, Any]
# Returns: {"input": {...}, "expected_output": str, "metadata": {...}}
```

### 4. Dataset Manager ([dataset_manager.py](dataset_manager.py))

**Upload Dataset:** `upload_finder_dataset(config, langfuse_client)`

Workflow:
1. Load FinDER dataset via `FinDERLoader`
2. Filter queries based on criteria (reasoning_required, category, etc.)
3. Transform queries to Langfuse items
4. Create dataset in Langfuse
5. Upload items in batches with rate limiting
6. Return upload summary

**Filter Options:**
- `ALL` - All queries
- `REASONING_REQUIRED` - Only queries requiring multi-step reasoning
- `NO_REASONING` - Simple lookup queries
- `BY_CATEGORY` - Filter by financial subdomain (equity, bonds, etc.)
- `BY_QUERY_TYPE` - Filter by query type (definition, comparison, etc.)
- `BY_COMPLEXITY` - Filter by complexity range

### 5. Configuration ([config.py](config.py))

**Credential Validation:** `validate_langfuse_credentials()`
- Checks for `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` in environment
- Raises `ValueError` if missing

**Config Loading:** `load_langfuse_config(config_path)`
- Loads system defaults from `src/config/langfuse.yaml`
- Provides flush intervals, batch sizes, retry policies

### 6. Utilities

**Exceptions ([exceptions.py](exceptions.py)):**
- `ExperimentExecutionError` - Task execution failures
- `DatasetNotFoundError` - Dataset loading failures
- `EvaluatorRegistrationError` - Invalid evaluator names
- `ConfigurationError` - Config validation failures

**Retry Logic ([retry.py](retry.py)):**
- `retry_with_backoff()` - Exponential backoff decorator
- Used for Langfuse API calls to handle rate limits

---

## Usage Examples

### Running an Experiment

```python
from src.langfuse_integration.experiment_runner import run_langfuse_experiment
from src.config.experiments import LangfuseExperimentConfig

# Load config from YAML
config = LangfuseExperimentConfig.from_yaml("experiments/configs/langfuse_baseline.yaml")

# Run experiment
result = run_langfuse_experiment(config)

# Access results
print(f"Success rate: {result.get_success_rate()}")
print(f"Average F1: {result.aggregate_scores['average_accuracy'].value}")
```

### Uploading a Dataset

```python
from src.langfuse_integration.dataset_manager import upload_finder_dataset
from src.langfuse_integration.models import LangfuseDatasetConfig, FilterCriteria
from langfuse import Langfuse

client = Langfuse()

config = LangfuseDatasetConfig(
    dataset_name="financial_qa_benchmark_v1",
    filter_criteria=FilterCriteria.ALL,
    ...
)

summary = upload_finder_dataset(config, client)
print(f"Uploaded {summary['total_items']} items")
```

### Custom Evaluator

```python
from langfuse.api.resources.commons import Evaluation

def evaluate_custom_metric(*, input, output, expected_output, metadata, **kwargs):
    # Your custom logic
    score = calculate_score(output, expected_output)

    return Evaluation(
        name="custom_metric",
        value=round(score, 4),
        data_type="NUMERIC",
        comment=f"Calculated using custom logic"
    )

# Register in evaluators.py:
item_evaluator_map = {
    "custom_metric": evaluate_custom_metric,
    ...
}
```

---

## Dependencies

### Internal
- `src.rag.baseline` - BaselineRAG pipeline
- `src.retrieval.*` - Retriever implementations
- `src.data_handler.loader` - FinDERLoader
- `src.data_handler.models` - Query, EvidencePassage
- `src.utils.llm_client` - LLM client factory
- `src.config.experiments` - Config schemas

### External
- `langfuse` - Langfuse SDK (required)
- `pydantic` - Data validation
- `sentence-transformers` - Semantic similarity (optional)
- `numpy` - Array operations for embeddings

---

## Current Status

✅ **Actively Used** - Core experiment execution workflow

All components are production-ready and actively used in the main workflow via [scripts/run_experiment.py](../../scripts/run_experiment.py).

---

## Configuration Example

```yaml
# experiments/configs/langfuse_baseline.yaml
name: "baseline-bm25-gpt35"
pipeline_type: "baseline"

langfuse_dataset_name: "financial_qa_benchmark_v1"
flush_interval: 5.0
flush_batch_size: 100
enable_tracing: true
max_concurrency: 1

item_evaluator_names:
  - "token_f1"
  - "retrieval_precision"
  - "retrieval_recall"

run_evaluator_names:
  - "average_accuracy"
  - "aggregate_retrieval_metrics"

evaluation_thresholds:
  token_f1: 0.3
```

---

## Key Design Decisions

1. **Closure Pattern for Task Functions**: Index and LLM client are initialized once and captured in closure, avoiding repeated initialization per query

2. **Langfuse SDK Integration**: Uses `dataset.run_experiment()` for parallel execution, automatic tracing, and evaluation aggregation

3. **Evaluator Separation**: Item-level evaluators run per query (fine-grained metrics), run-level evaluators aggregate across all queries (overall performance)

4. **Pydantic Models**: Type-safe result objects with validation and helper methods

5. **Batch Upload**: Dataset upload uses batching and rate limiting to avoid API limits

---

## See Also

- Main Architecture: [../../ARCHITECTURE.md](../../ARCHITECTURE.md)
- RAG Pipeline: [../rag/CLAUDE.md](../rag/CLAUDE.md)
- Configuration: [../config/CLAUDE.md](../config/CLAUDE.md)
- Entry Point: [../../scripts/run_experiment.py](../../scripts/run_experiment.py)
