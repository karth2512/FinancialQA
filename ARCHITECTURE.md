# FinancialQA System Architecture

## Executive Summary

FinancialQA is a **config-driven RAG (Retrieval-Augmented Generation) system** for financial question-answering using the FinDER dataset. The system integrates with **Langfuse SDK** for comprehensive experiment tracking, evaluation, and dataset management.

**Current Status:** Production-ready RAG pipeline with BM25 retrieval and two pipeline types (baseline, query expansion). GraphRAG retrieval code exists but requires missing `src/graphrag_models/` module to function.

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENTRY POINT                                  │
│              scripts/run_experiment.py                          │
│        (CLI interface + configuration loading)                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              EXPERIMENT ORCHESTRATION                           │
│    src/langfuse_integration/experiment_runner.py               │
│  (Create task, register evaluators, run via Langfuse SDK)      │
└──────┬──────────────────────────────┬───────────────────────────┘
       │                              │
       ▼                              ▼
┌──────────────────────────┐  ┌─────────────────────────┐
│   RAG PIPELINES          │  │   EVALUATION SYSTEM     │
│ src/rag/                 │  │ src/langfuse_integ/     │
│                          │  │   evaluators.py         │
│ - baseline.py            │  │                         │
│ - query_expansion.py     │  │ Item-level + Run-level  │
│                          │  │ Evaluators              │
└──────┬───────────────────┘  └─────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│                  CORE COMPONENTS                                 │
├─────────────────┬───────────────────┬──────────────┬────────────┤
│   RETRIEVAL     │   DATA HANDLER    │  CONFIG      │  UTILS     │
│ src/retrieval/  │ src/data_handler/ │ src/config/  │ src/utils/ │
│                 │                   │              │            │
│ - base.py       │ - loader.py       │ - exper.py   │ - llm_...  │
│ - bm25.py       │ - models.py       │              │ - cache.py │
│ - dense.py      │ - indexer.py      │              │            │
│ - hybrid.py     │                   │              │            │
│ - graphrag.py   │                   │              │            │
└─────────────────┴───────────────────┴──────────────┴────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  LANGFUSE SDK        │
                  │  (External Service)  │
                  │  - Dataset mgmt      │
                  │  - Experiment exec   │
                  │  - Tracing & eval    │
                  └──────────────────────┘
```

---

## Project Structure

```
src/
├── langfuse_integration/  # Langfuse SDK wrapper + experiment logic (CORE)
│   ├── __init__.py
│   ├── experiment_runner.py  # Main experiment orchestration
│   ├── evaluators.py         # Item & run-level evaluators
│   ├── models.py             # Pydantic models for results
│   ├── config.py             # Langfuse config helpers
│   ├── dataset_manager.py    # Dataset upload to Langfuse
│   ├── exceptions.py         # Custom exceptions
│   └── retry.py              # Retry utilities
│
├── rag/                     # RAG pipeline implementations
│   ├── __init__.py
│   ├── baseline.py          # BaselineRAG: retrieve → generate pattern (CORE)
│   └── query_expansion.py   # QueryExpansionRAG: expand → retrieve → generate (CORE)
│
├── retrieval/               # Passage retrieval strategies
│   ├── __init__.py
│   ├── base.py              # RetrieverBase abstract class
│   ├── bm25.py              # BM25 retrieval (ACTIVELY USED)
│   ├── dense.py             # Dense retrieval (NOT actively tested)
│   ├── hybrid.py            # Hybrid retrieval (NOT actively tested)
│   └── graphrag.py          # GraphRAG retrieval (REQUIRES MISSING graphrag_models)
│
├── data_handler/            # Dataset loading & preprocessing
│   ├── __init__.py
│   ├── models.py            # Data classes (Query, EvidencePassage, etc.)
│   ├── loader.py            # FinDERLoader: Load from HuggingFace
│   └── indexer.py           # Build retrieval indexes
│
├── config/                  # Configuration management
│   ├── __init__.py
│   └── experiments.py       # ExperimentConfig & LangfuseExperimentConfig (Pydantic)
│
├── utils/                   # Shared utilities
│   ├── __init__.py
│   ├── llm_client.py        # LLM abstraction (OpenAI, Anthropic, local)
│   └── cache.py             # Caching utilities
scripts/
├── run_experiment.py         # Main entry point for experiments
├── upload_dataset.py         # Upload dataset to Langfuse
├── compare_experiments.py    # Compare experiment results
├── validate_setup.py         # Validate environment setup
├── convert_arrow_to_text.py  # Convert arrow files to text
├── prepare_graphrag_input.py # Copy pre-chunked text files for GraphRAG
├── configure_graphrag.py     # Auto-configure GraphRAG settings
└── inspect_graphrag_index.py # Inspect parquet outputs

experiments/configs/          # Experiment YAML configs
├── dev.yaml                  # Development config
├── baseline.yaml             # Basic baseline config
├── langfuse_baseline.yaml    # BM25 + Claude baseline
├── langfuse_denseret.yaml    # Dense retrieval config
├── query_expansion.yaml      # Query expansion with BM25
├── graphrag_baseline.yaml    # GraphRAG local search + baseline
├── graphrag_global.yaml      # GraphRAG global search + baseline
└── graphrag_query_expansion.yaml  # GraphRAG + query expansion

data/                         # Data directory (gitignored)
└── finder/                   # FinDER dataset cache
```

---

## Data Flow

### Complete Experiment Execution Flow

```
1. CONFIGURATION LOADING
   run_experiment.py
   ├─ Load .env environment variables
   ├─ Parse CLI arguments (config_path, --dry-run, etc.)
   ├─ LangfuseExperimentConfig.from_yaml(config_path)
   │    └─ Pydantic validation (ExperimentConfig schema)
   └─ Validate Langfuse credentials

2. LANGFUSE INITIALIZATION
   experiment_runner.py
   ├─ Create Langfuse() client with flush settings
   └─ Load dataset from Langfuse
       └─ Returns DatasetClient with .items list

3. TASK FUNCTION CREATION (Closure Pattern)
   create_task_function(config)
   ├─ Load corpus from FinDERLoader
   ├─ Create retriever based on strategy:
   │   ├─ "bm25" → BM25Retriever (WORKING)
   │   ├─ "dense" → DenseRetriever (NOT TESTED)
   │   ├─ "hybrid" → HybridRetriever (NOT TESTED)
   │   └─ "graphrag" → GraphRAGRetriever (MISSING DEPENDENCY)
   ├─ retriever.index_corpus(corpus)
   ├─ Create LLM client(s)
   ├─ Create RAG pipeline:
   │   ├─ "baseline" → BaselineRAG
   │   └─ "query_expansion" → QueryExpansionRAG
   └─ Return task function (closure captures initialized components)

4. EVALUATOR REGISTRATION
   register_evaluators()
   ├─ Load item_evaluator_names from config
   │    └─ token_f1, semantic_sim, retrieval_precision, etc.
   └─ Load run_evaluator_names from config
        └─ average_accuracy, aggregate_retrieval_metrics, pass_rate

5. EXPERIMENT EXECUTION
   dataset.run_experiment()
   ├─ For each dataset item:
   │   ├─ task_function(item=item)
   │   │   ├─ RAG pipeline processes query
   │   │   │   ├─ [Query Expansion] Generate M query variants
   │   │   │   ├─ Retrieve passages
   │   │   │   ├─ [Query Expansion] Pool and deduplicate passages
   │   │   │   └─ Generate answer via LLM
   │   │   └─ Return {answer, passages, latency, cost}
   │   └─ Run item-level evaluators
   │       ├─ evaluate_token_f1()
   │       ├─ evaluate_semantic_similarity()
   │       ├─ evaluate_retrieval_precision()
   │       └─ evaluate_retrieval_recall()
   └─ Run run-level evaluators (aggregate)
       ├─ compute_average_accuracy()
       ├─ compute_aggregate_retrieval_metrics()
       └─ PassRateEvaluator()

6. RESULT TRANSFORMATION
   _transform_experiment_result()
   ├─ Extract item results from SDK result
   ├─ Convert SDK ItemResult → ItemExecutionResult
   ├─ Extract aggregate scores
   └─ Return ExperimentRunOutput

7. RESULT DISPLAY
   run_experiment.py (main)
   ├─ Display results summary
   ├─ Write to JSON if --output specified
   └─ Exit with status code
```

---

## RAG Pipeline Implementations

### 1. Baseline RAG ([src/rag/baseline.py](src/rag/baseline.py))

**Single-stage pipeline**:
```
Query → Retrieve → Generate
```

**Components**:
- Retriever: BM25/Dense/Hybrid
- Generator: Single LLM

**Use Case**: Fast, simple RAG baseline

---

### 2. Query Expansion RAG ([src/rag/query_expansion.py](src/rag/query_expansion.py))

**Multi-query pipeline**:
```
Query → Expand (M variants) → Retrieve (M+1 times) → Pool & Deduplicate → Generate
```

**Components**:
- Expander: LLM for query reformulation (higher temperature for diversity)
- Retriever: BM25/Dense/Hybrid (reused M+1 times)
- Generator: LLM for answer synthesis (zero temperature for reproducibility)

**Architecture**:
1. **Expansion**: Generate M query variants via expander LLM with structured output (Pydantic `ExpandedQueries` model)
2. **Retrieval**: Retrieve top_k passages for each of M+1 queries
3. **Pooling**: Deduplicate by passage ID, keep highest scores
4. **Re-ranking**: Sort pooled passages by score (descending)
5. **Generation**: Use pooled passages for final answer

**Use Case**: Improved recall when single query misses relevant passages

**Trade-offs**:
- Recall: +10-25% (query diversity finds more relevant passages)
- Precision: -5-10% (some expanded queries less relevant)
- Latency: +50-100% (expansion LLM + M+1 retrievals)
- Cost: +100-200% (expansion tokens + more retrieval)

---

## Retrieval Strategies

### BM25 Retrieval ([src/retrieval/bm25.py](src/retrieval/bm25.py)) - **ACTIVELY USED**

**Library**: rank_bm25 (BM25Okapi algorithm)

**How it works**:
1. **Indexing**: Tokenize corpus with whitespace splitting
2. **Retrieval**: Score documents by BM25 formula (TF-IDF variant)
3. **Ranking**: Return top-k by score

**Configuration**:
```yaml
retrieval_config:
  strategy: "bm25"
  top_k: 5
  k1: 1.5  # Term frequency saturation
  b: 0.75  # Length normalization
```

**Performance**: <0.1s per query, free (local computation)

---

### Dense Retrieval ([src/retrieval/dense.py](src/retrieval/dense.py)) - **NOT ACTIVELY TESTED**

**Library**: sentence-transformers + ChromaDB

**How it works**: Semantic similarity via embeddings

---

### Hybrid Retrieval ([src/retrieval/hybrid.py](src/retrieval/hybrid.py)) - **NOT ACTIVELY TESTED**

**How it works**: Weighted combination of BM25 + Dense scores

---

### GraphRAG Retrieval ([src/retrieval/graphrag.py](src/retrieval/graphrag.py)) - **INCOMPLETE**

**Status**: Code exists but requires missing `src/graphrag_models/` module containing:
- `AnthropicChatModel` - Chat model for entity extraction
- `HFEmbeddingModel` - Embedding model for semantic search
- `register_models()` - Model registration function

The retriever expects these modules but they do not exist in the codebase. GraphRAG functionality will fail until these are implemented.

---

## Key Abstractions & Interfaces

### 1. Retriever Interface

**File:** [src/retrieval/base.py](src/retrieval/base.py)

```python
class RetrieverBase(ABC):
    """All retrieval implementations follow this contract"""

    @abstractmethod
    def index_corpus(self, passages: List[EvidencePassage]) -> None:
        """One-time indexing of evidence passages"""

    @abstractmethod
    def retrieve(self, query_text: str, top_k: int = 5) -> RetrievalResult:
        """Return ranked passages for query"""
```

**Implementations:**
- `BM25Retriever` - Keyword-based (rank_bm25 library) - **ACTIVELY USED**
- `DenseRetriever` - Semantic search (sentence-transformers + ChromaDB) - **NOT ACTIVELY TESTED**
- `HybridRetriever` - Combined with rank fusion - **NOT ACTIVELY TESTED**
- `GraphRAGRetriever` - Knowledge graph-based - **MISSING DEPENDENCIES**

### 2. LLM Client Interface

**File:** [src/utils/llm_client.py](src/utils/llm_client.py)

```python
class LLMClient(ABC):
    """All LLM providers implement this interface"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Estimate token count"""

    @abstractmethod
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost in USD"""
```

**Implementations:**
- `OpenAIClient` - GPT-3.5, GPT-4 with retry logic
- `AnthropicClient` - Claude models with retry logic
- `LocalModelClient` - HuggingFace transformers (offline)

### 3. Data Models

**File:** [src/data_handler/models.py](src/data_handler/models.py)

```
Query
  ├─ id: str
  ├─ text: str
  ├─ expected_answer: str
  ├─ expected_evidence: List[str]
  └─ metadata: QueryMetadata

EvidencePassage
  ├─ id: str
  ├─ text: str
  ├─ document_id: str
  └─ metadata: EvidenceMetadata

RetrievalResult
  ├─ query_id: str
  ├─ retrieved_passages: List[RetrievedPassage]
  ├─ strategy: str  # "bm25", "dense", "hybrid", "graphrag_local", "graphrag_global"
  ├─ retrieval_time_seconds: float
  └─ metadata: Dict
```

### 4. Configuration Schema

**File:** [src/config/experiments.py](src/config/experiments.py)

```
LangfuseExperimentConfig (extends ExperimentConfig)
  ├─ name: str
  ├─ description: str
  ├─ run_id: str
  ├─ pipeline_type: str ("baseline" | "query_expansion")
  ├─ retrieval_config: RetrievalConfig
  │   ├─ strategy: str ("bm25" | "dense" | "hybrid" | "graphrag")
  │   ├─ top_k, k1, b
  │   └─ graphrag_root, graphrag_method, graphrag_community_level (if graphrag)
  ├─ llm_configs: Dict[str, LLMConfig]
  │   ├─ generator (required for all)
  │   └─ expander (required for query_expansion)
  ├─ hyperparameters: Dict[str, Any]
  │   └─ num_expanded_queries (for query_expansion)
  ├─ langfuse_dataset_name: Optional[str]
  ├─ max_concurrency: int (1-20)
  ├─ item_evaluator_names: List[str]
  ├─ run_evaluator_names: List[str]
  └─ evaluation_thresholds: Dict[str, float]
```

---

## Module Dependencies

### Import Graph

```
scripts/run_experiment.py
  ├─ src.langfuse_integration.experiment_runner
  ├─ src.config.experiments
  └─ src.langfuse_integration.config

src/langfuse_integration/experiment_runner.py
  ├─ src.rag.baseline (BaselineRAG)
  ├─ src.rag.query_expansion (QueryExpansionRAG)
  ├─ src.retrieval.bm25, .dense, .hybrid, .graphrag
  ├─ src.data_handler.loader (FinDERLoader)
  ├─ src.data_handler.models (Query, EvidencePassage)
  ├─ src.utils.llm_client (create_llm_client)
  ├─ src.langfuse_integration.evaluators
  ├─ src.langfuse_integration.models
  └─ src.config.experiments

src/retrieval/graphrag.py
  ├─ src.retrieval.base (RetrieverBase)
  ├─ src.data_handler.models (EvidencePassage, RetrievalResult)
  ├─ src.graphrag_models (MISSING - AnthropicChatModel, HFEmbeddingModel)
  └─ External: pandas
```

**Note:** GraphRAG retriever has broken import dependency on non-existent `src.graphrag_models` module.

---

## Configuration System

### How YAML Drives Execution

```yaml
# experiments/configs/langfuse_baseline.yaml
name: "bm25_baseline_v1"
run_id: "exp-001"
pipeline_type: "baseline"

retrieval_config:
  strategy: "bm25"       # → BM25Retriever
  top_k: 5               # → Number of passages to retrieve
  k1: 1.5                # → BM25 term frequency saturation
  b: 0.75                # → BM25 length normalization

llm_configs:
  generator:
    provider: "anthropic"      # → Which LLM client to use
    model: "claude-3-5-haiku-20241022"
    temperature: 0.0

langfuse_dataset_name: "financial_qa_benchmark_v1"
max_concurrency: 1
```

### Configuration Flow

```
YAML File
  ↓
LangfuseExperimentConfig.from_yaml()
  ↓ (Pydantic validation)
Validated Config Object
  ↓
create_task_function(config)
  ├─ config.retrieval_config.strategy → BM25Retriever/etc.
  ├─ config.pipeline_type → BaselineRAG/QueryExpansionRAG
  ├─ config.llm_configs["generator"] → OpenAIClient/AnthropicClient
  └─ All become closures in task function
```

---

## Critical Implementation Details

### 1. Task Function Creation (Closure Pattern)

The task function is a **closure** that captures initialized components:

```python
def _create_baseline_task(config):
    # Initialize once (closure captures these)
    corpus = loader.load_corpus()
    retriever = BM25Retriever(config.retrieval_config)
    retriever.index_corpus(corpus)
    llm_client = create_llm_client(config.llm_configs["generator"])
    rag_pipeline = BaselineRAG(retriever, llm_client)

    # This function is called once per dataset item
    def baseline_task(*, item, **kwargs):
        query = extract_from_item(item)
        rag_result = rag_pipeline.process_query(query, top_k=5)
        return {
            "answer": rag_result["generated_answer"],
            "retrieved_passages": [...],
            "latency_seconds": rag_result["latency_seconds"],
            ...
        }

    return baseline_task
```

**Key Point:** Index & LLM client are initialized ONCE, reused for all items.

### 2. Evaluator Pattern

```python
# Item-level: Called once per item after task executes
def evaluate_token_f1(*, input, output, expected_output, metadata, **kwargs):
    # Calculate F1 between generated and expected answer
    return Evaluation(name="token_f1", value=f1, data_type="NUMERIC")

# Run-level: Called once after all items processed
def compute_average_accuracy(*, item_results, **kwargs):
    # Average F1 across all queries
    return Evaluation(name="average_accuracy", value=avg, data_type="NUMERIC")
```

---

## Experiment Configurations

### Available Configs

| Config File | Pipeline | Retrieval | Status |
|-------------|----------|-----------|--------|
| `langfuse_baseline.yaml` | baseline | BM25 | Working |
| `query_expansion.yaml` | query_expansion | BM25 | Working |
| `graphrag_baseline.yaml` | baseline | GraphRAG local | Missing dependencies |
| `graphrag_global.yaml` | baseline | GraphRAG global | Missing dependencies |
| `graphrag_query_expansion.yaml` | query_expansion | GraphRAG local | Missing dependencies |

---

## Extension Points

### Adding New Retrieval Strategy

1. Create new class in [src/retrieval/](src/retrieval/) implementing `RetrieverBase`
2. Add to strategy validation in [experiments.py](src/config/experiments.py)
3. Update `_create_baseline_task()` in [experiment_runner.py](src/langfuse_integration/experiment_runner.py)

### Adding New LLM Provider

1. Create new client in [llm_client.py](src/utils/llm_client.py) implementing `LLMClient`
2. Add to provider validation in [experiments.py](src/config/experiments.py)
3. Update `create_llm_client()` factory

### Adding New Evaluator

1. Implement in [evaluators.py](src/langfuse_integration/evaluators.py)
2. Register in `register_evaluators()`
3. Allow in config validation

### Adding New Pipeline Type

1. Create new class in [src/rag/](src/rag/) implementing `process_query()` interface
2. Add pipeline_type to validator in [experiments.py](src/config/experiments.py)
3. Add task creation function in [experiment_runner.py](src/langfuse_integration/experiment_runner.py)
4. Create example config in [experiments/configs/](experiments/configs/)

---

## Current Status & Limitations

### Supported
- Baseline RAG pipeline with BM25 retrieval
- Query Expansion RAG pipeline with dual LLM setup
- OpenAI/Anthropic LLM integration
- Langfuse experiment tracking
- Item and run-level evaluation
- Config-driven experiments

### Partially Supported
- Dense/hybrid retrieval (implementations exist but untested)
- Local LLM models (basic implementation)

### Not Supported / Broken
- GraphRAG retrieval (missing `src/graphrag_models/` module)
- Local data loading (marked TODO in experiment_runner.py)
- Reranking (config option exists but not implemented)

---

## Key Workflows

### 1. Dataset Upload (One-Time)

```bash
python scripts/upload_dataset.py --name financial_qa_benchmark_v1 --filter all
```

Flow: Load FinDER → Filter queries → Transform to Langfuse items → Upload in batches

### 2. Experiment Execution

```bash
python scripts/run_experiment.py experiments/configs/langfuse_baseline.yaml
```

Flow: Load config → Create task → Register evaluators → Execute via Langfuse SDK → Display results

### 3. Experiment Comparison

```bash
python scripts/compare_experiments.py exp-001 exp-002 exp-003
```

---

## External Dependencies

### Required
- `langfuse` - Experiment tracking SDK
- `pydantic` - Configuration validation
- `rank-bm25` - BM25 retrieval
- `datasets` - HuggingFace FinDER dataset
- `openai` or `anthropic` - LLM providers
- `python-dotenv` - Environment variables
- `pyyaml` - Configuration loading
- `numpy` - Vector operations

### Optional
- `chromadb` - Vector storage for dense retrieval
- `sentence-transformers` - Local embeddings
- `transformers` - Local LLM models
- `pandas` - GraphRAG parquet file handling

---

## References

- **Main CLAUDE.md**: [CLAUDE.md](CLAUDE.md)
