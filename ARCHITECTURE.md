# FinancialQA System Architecture

## Executive Summary

FinancialQA is a **config-driven RAG (Retrieval-Augmented Generation) system** for financial question-answering using the FinDER dataset. The system integrates with **Langfuse SDK** for comprehensive experiment tracking, evaluation, and dataset management.

**Current Status:** Production-ready baseline RAG pipeline with BM25 retrieval and Langfuse integration. The codebase has been simplified (Jan 2026) to focus on core RAG workflows.

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENTRY POINT                                  │
│              scripts/run_experiment.py                           │
│        (CLI interface + configuration loading)                   │
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
┌──────────────────┐        ┌─────────────────────────┐
│   RAG PIPELINE   │        │   EVALUATION SYSTEM     │
│ src/rag/baseline │        │ src/langfuse_integ/     │
│                  │        │   evaluators.py         │
│ - Retriever      │        │                         │
│ - LLM Generate   │        │ Item-level + Run-level  │
│ - Result format  │        │ Evaluators              │
└──────┬───────────┘        └─────────────────────────┘
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
│   ├── experiment_runner.py  # Main experiment orchestration
│   ├── evaluators.py         # Item & run-level evaluators
│   ├── models.py             # Pydantic models for results
│   ├── config.py             # Langfuse config helpers
│   ├── dataset_manager.py    # Dataset upload to Langfuse
│   ├── exceptions.py         # Custom exceptions
│   └── retry.py              # Retry utilities
│
├── rag/                     # RAG pipeline implementations
│   ├── baseline.py          # BaselineRAG: retrieve → generate pattern (CORE)
│   └── query_expansion.py   # QueryExpansionRAG: expand → retrieve → generate (NEW)
│
├── retrieval/               # Passage retrieval strategies
│   ├── base.py              # RetrieverBase abstract class
│   ├── bm25.py              # BM25 retrieval (ACTIVELY USED)
│   ├── dense.py             # Dense retrieval (NOT actively tested)
│   ├── hybrid.py            # Hybrid retrieval (NOT actively tested)
│   └── graphrag.py          # GraphRAG knowledge graph retrieval (NEW)
│
├── data_handler/            # Dataset loading & preprocessing
│   ├── models.py            # Data classes (Query, EvidencePassage, etc.)
│   ├── loader.py            # FinDERLoader: Load from HuggingFace
│   └── indexer.py           # Build retrieval indexes
│
├── evaluation/              # Legacy evaluation system (pre-Langfuse)
│   ├── metrics.py           # Metric calculators
│   ├── models.py            # Evaluation data structures
│   └── logger.py            # Result logging
│
├── config/                  # Configuration management
│   └── experiments.py       # ExperimentConfig & LangfuseExperimentConfig (Pydantic)
│
└── utils/                   # Shared utilities
    ├── llm_client.py        # LLM abstraction (OpenAI, Anthropic, local)
    └── cache.py             # Caching utilities

scripts/
├── run_experiment.py         # Main entry point for experiments
├── upload_dataset.py         # Upload dataset to Langfuse
└── compare_experiments.py    # Compare experiment results

experiments/configs/          # Experiment YAML configs
data/                         # Data directory (gitignored)
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

3. TASK FUNCTION CREATION
   create_task_function(config)
   ├─ Load corpus from FinDERLoader
   ├─ Create retriever (BM25/dense/hybrid)
   ├─ retriever.index_corpus(corpus)
   ├─ Create LLM client
   ├─ Create BaselineRAG(retriever, llm_client)
   └─ Return baseline_task function

4. EVALUATOR REGISTRATION
   register_evaluators()
   ├─ Load item_evaluator_names from config
   │    └─ token_f1, semantic_sim, retrieval_precision, etc.
   └─ Load run_evaluator_names from config
        └─ average_accuracy, aggregate_retrieval_metrics, pass_rate

5. EXPERIMENT EXECUTION
   dataset.run_experiment()
   ├─ For each dataset item:
   │   ├─ baseline_task(item=item)
   │   │   ├─ BaselineRAG.process_query(query)
   │   │   │   ├─ retriever.retrieve(query_text, top_k)
   │   │   │   ├─ Construct prompt from query + passages
   │   │   │   ├─ llm_client.generate(prompt)
   │   │   │   └─ Return {answer, passages, latency, cost}
   │   │   └─ Return task result
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
- Expander: LLM for query reformulation (e.g., GPT-3.5)
- Retriever: BM25/Dense/Hybrid (reused M+1 times)
- Generator: LLM for answer synthesis (e.g., Claude)

**Architecture**:
1. **Expansion**: Generate M query variants via expander LLM with structured output (Pydantic)
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

### 3. GraphRAG Retrieval ([src/retrieval/graphrag.py](src/retrieval/graphrag.py))

**Knowledge graph-based retrieval**:
```
Documents → Entity Extraction → Relationship Detection → Community Clustering → Indexed Graph
Query → Graph Traversal (Local/Global) → Relevant Subgraph → Retrieved Passages
```

**Components**:
- Indexing: Microsoft GraphRAG library with Anthropic Claude + local embeddings
- Retriever: GraphRAGRetriever with local (entity-focused) and global (community-level) search
- Generator: Single LLM (works with both baseline and query_expansion pipelines)

**Architecture**:

**One-Time Indexing** (via `scripts/graphrag_index.py`):
1. **Chunking**: Split 5,703 text files into 300-token chunks
2. **Entity Extraction**: Anthropic Claude 3.5 Haiku extracts financial entities (organizations, metrics, regulations)
3. **Relationship Detection**: Claude identifies relationships between entities
4. **Embedding**: sentence-transformers/all-MiniLM-L6-v2 (384d, local, no API)
5. **Community Detection**: NetworkX hierarchical clustering (levels 0-5)
6. **Community Summarization**: Claude generates summaries for each community
7. **Output**: Parquet files in `data/graphrag/output/` (~100-200 MB)

**Query-Time Retrieval** (two methods):

**Local Search** (`graphrag_method: "local"`):
- Entity-focused retrieval
- Finds entities matching query keywords
- Retrieves entity neighborhoods (1-2 hops)
- Best for: Specific questions about known entities
- Example: "What is Apple's revenue?" → Retrieves Apple entity + financial metrics

**Global Search** (`graphrag_method: "global"`):
- Community-level retrieval
- Retrieves pre-computed community summaries
- Hierarchical (uses `community_level` parameter)
- Best for: Broad questions requiring multiple entities
- Example: "What are tech industry trends?" → Retrieves tech sector community summaries

**Use Cases**:
- Multi-hop reasoning (e.g., "Compare revenues of companies in same sector")
- Entity disambiguation (e.g., distinguishing "Apple Inc." from "Apple Records")
- Relationship-aware retrieval (e.g., "Who are Tesla's competitors?")
- Financial domain with many interconnected entities

**Configuration**:
```yaml
retrieval_config:
  strategy: "graphrag"
  top_k: 5
  graphrag_root: "./data/graphrag"
  graphrag_method: "local"  # or "global"
  graphrag_community_level: 2  # for global search (0-5)
  graphrag_response_type: "Multiple Paragraphs"
```

**Trade-offs**:

Advantages:
- Entity-aware retrieval (understands "Apple Inc." vs. "Apple Records")
- Multi-hop reasoning support (traverse relationships)
- Better for complex financial queries with entity relationships
- Pre-computed communities for fast global search
- No query-time embedding needed (graph traversal only)

Disadvantages:
- High one-time indexing cost (~$3-6 USD for Anthropic Claude)
- Slow indexing (30-90 minutes for 5,703 documents)
- Higher query latency (1-5s vs. <0.1s for BM25)
- Requires pre-built index (cannot handle new documents at query time)
- More complex setup (indexing script, multiple parquet files)

**Indexing Performance**:
- Time: 30-90 minutes (depends on local hardware for embeddings)
- Cost: ~$3-6 USD (Anthropic Claude only; embeddings free via sentence-transformers)
- Output Size: ~100-200 MB parquet files
- Memory: ~500MB (embedding model) + ~1GB (GraphRAG processing)

**Query Performance**:
- Local Search: 1-3s per query (~$0.001-0.002 USD for generation)
- Global Search: 3-5s per query (~$0.003-0.005 USD for generation)
- BM25 Baseline: <0.1s per query (free)

**Setup Requirements**:
1. Install dependencies: `pip install graphrag anthropic sentence-transformers`
2. Set Anthropic API key: `export ANTHROPIC_API_KEY=sk-ant-...`
3. Run indexing: `python scripts/graphrag_index.py` (30-90 min, ~$3-6 USD)
4. Verify output: `ls data/graphrag/output/*.parquet`
5. Run experiments: Use config files in `experiments/configs/graphrag_*.yaml`

**Integration with Pipelines**:
- **Baseline + GraphRAG**: `experiments/configs/graphrag_baseline.yaml`
- **Global Search**: `experiments/configs/graphrag_global.yaml`
- **Query Expansion + GraphRAG**: `experiments/configs/graphrag_query_expansion.yaml`

GraphRAG is implemented as a **retrieval strategy** (like bm25, dense, hybrid), so it works seamlessly with existing baseline and query_expansion pipelines. This allows direct comparison of BM25 vs. GraphRAG while keeping all other variables constant.

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
- `GraphRAGRetriever` - Knowledge graph-based (Microsoft GraphRAG) - **NEW**

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
  ├─ strategy: str
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
  ├─ pipeline_type: str (only "baseline" valid)
  ├─ retrieval_config: RetrievalConfig
  ├─ llm_configs: Dict[str, LLMConfig]
  ├─ langfuse_dataset_name: Optional[str]
  ├─ max_concurrency: int (1-20)
  ├─ enable_item_evaluators: bool
  ├─ enable_run_evaluators: bool
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
  ├─ src.retrieval.bm25, .dense, .hybrid
  ├─ src.data_handler.loader (FinDERLoader)
  ├─ src.data_handler.models (Query, EvidencePassage)
  ├─ src.utils.llm_client (create_llm_client)
  ├─ src.langfuse_integration.evaluators
  ├─ src.langfuse_integration.models
  └─ src.config.experiments

src/rag/baseline.py
  ├─ src.retrieval.base (RetrieverBase)
  ├─ src.data_handler.models (Query, RetrievalResult)
  └─ src.utils.llm_client (LLMClient)

src/retrieval/{bm25,dense,hybrid}.py
  ├─ src.retrieval.base (RetrieverBase)
  └─ src.data_handler.models (EvidencePassage, RetrievalResult)

src/data_handler/loader.py
  ├─ src.data_handler.models (Query, EvidencePassage)
  └─ External: datasets (HuggingFace)

src/utils/llm_client.py
  └─ External: openai, anthropic, transformers
```

**Note:** No circular dependencies - clean DAG structure.

---

## Configuration System

### How YAML Drives Execution

```yaml
# experiments/configs/langfuse_baseline.yaml
name: "baseline-bm25-gpt35"
run_id: "exp-001"
pipeline_type: "baseline"

retrieval_config:
  strategy: "bm25"    # → Determines which retriever is instantiated
  top_k: 5            # → Number of passages to retrieve
  k1: 1.5             # → BM25 hyperparameter
  b: 0.75             # → BM25 hyperparameter

llm_configs:
  generator:
    provider: "openai"      # → Which LLM client to use
    model: "gpt-3.5-turbo"  # → Which model to call
    temperature: 0.0        # → Generation sampling

langfuse_dataset_name: "financial_qa_benchmark_v1"
max_concurrency: 1

item_evaluator_names:
  - "token_f1"
  - "retrieval_precision"
  - "retrieval_recall"

run_evaluator_names:
  - "average_accuracy"
  - "aggregate_retrieval_metrics"
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
  ├─ config.retrieval_config.strategy → BM25Retriever/DenseRetriever/HybridRetriever
  ├─ config.retrieval_config.top_k → retriever.retrieve(top_k)
  ├─ config.llm_configs["generator"] → OpenAIClient/AnthropicClient/LocalModelClient
  └─ All become closures in task function
```

---

## Component Relationships

### Core Pipeline Components

```
EXPERIMENT RUNNER (experiment_runner.py)
  │
  ├─ RAG MODULE (baseline.py)
  │   ├─ RETRIEVER (bm25.py/dense.py/hybrid.py)
  │   │   └─ DATA HANDLER (loader.py)
  │   └─ LLM CLIENT (llm_client.py)
  │
  └─ EVALUATORS MODULE (evaluators.py)
      └─ METRICS (evaluation/metrics.py)
```

---

## Critical Implementation Details

### 1. Task Function Creation

The task function is a **closure** that captures initialized components:

```python
def _create_baseline_task(config):
    # Initialize once (closure captures these)
    corpus = loader.load_corpus()
    retriever = BM25Retriever(config.retrieval_config)
    retriever.index_corpus(corpus)  # Build index once
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

### 2. BaselineRAG Pipeline

**File:** [src/rag/baseline.py:35-84](src/rag/baseline.py#L35-L84)

For each query:
1. **RETRIEVE** - Get top-k passages via retriever
2. **CONSTRUCT PROMPT** - Combine query + ranked passages
3. **COUNT TOKENS** - Estimate prompt tokens
4. **GENERATE** - Call LLM
5. **ESTIMATE COST** - Calculate based on tokens

### 3. BM25 Retrieval Pipeline

**File:** [src/retrieval/bm25.py:33-102](src/retrieval/bm25.py#L33-L102)

```python
def index_corpus(self, passages):
    tokenized_corpus = [p.text.lower().split() for p in passages]
    self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)

def retrieve(self, query_text, top_k):
    tokenized_query = query_text.lower().split()
    scores = self.bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i])[:top_k]
    return RetrievalResult(retrieved_passages=[...])
```

### 4. Evaluator Pattern

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

## Extension Points

### Adding New Retrieval Strategy

1. Create new class in [src/retrieval/](src/retrieval/)
2. Add to retrieval_config validation in [experiments.py](src/config/experiments.py)
3. Update `_create_baseline_task()` in [experiment_runner.py](src/langfuse_integration/experiment_runner.py)

### Adding New LLM Provider

1. Create new client in [llm_client.py](src/utils/llm_client.py)
2. Add to provider validation in [experiments.py](src/config/experiments.py)
3. Update `create_llm_client()` factory

### Adding New Evaluator

1. Implement in [evaluators.py](src/langfuse_integration/evaluators.py)
2. Register in `register_evaluators()`
3. Allow in config validation

### Adding New Pipeline Type

1. Create new class in [src/rag/](src/rag/) implementing process_query interface
2. Add pipeline_type to validator in [experiments.py](src/config/experiments.py)
3. Add task creation function in [experiment_runner.py](src/langfuse_integration/experiment_runner.py)
4. Create example config in [experiments/configs/](experiments/configs/)
5. Update documentation (CLAUDE.md, ARCHITECTURE.md, README.md)

**Example**: QueryExpansionRAG follows this pattern exactly.

---

## Current Status & Limitations

### Supported
- ✅ Baseline RAG pipeline with BM25 retrieval
- ✅ Query Expansion RAG pipeline with dual LLM setup
- ✅ GraphRAG retrieval with local and global search
- ✅ OpenAI/Anthropic LLM integration
- ✅ Langfuse experiment tracking
- ✅ Item and run-level evaluation
- ✅ Config-driven experiments

### Partially Supported
- ⚠️ Dense/hybrid retrieval (implementations exist but untested)
- ⚠️ Local LLM models (basic implementation)
- ⚠️ GraphRAG (indexing script and retriever implemented, needs testing with actual GraphRAG library)

### Not Supported
- ❌ Multi-agent pipelines (scaffolding removed)
- ❌ Local data loading (marked TODO)
- ❌ Answer verification
- ❌ Reranking (config option exists but not implemented)

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

### Optional
- `sentence-transformers` - Dense retrieval and GraphRAG embeddings
- `chromadb` - Vector storage for dense retrieval
- `transformers` - Local LLM models
- `graphrag` - Microsoft GraphRAG library for knowledge graph-based retrieval
- `anthropic` - Anthropic Claude API (required for GraphRAG indexing)

---

## References

- **Main CLAUDE.md**: [CLAUDE.md](CLAUDE.md)
- **Module Documentation**:
  - [src/langfuse_integration/CLAUDE.md](src/langfuse_integration/CLAUDE.md)
  - [src/data_handler/CLAUDE.md](src/data_handler/CLAUDE.md)
  - [src/retrieval/CLAUDE.md](src/retrieval/CLAUDE.md)
  - [src/rag/CLAUDE.md](src/rag/CLAUDE.md)
  - [src/evaluation/CLAUDE.md](src/evaluation/CLAUDE.md)
  - [src/config/CLAUDE.md](src/config/CLAUDE.md)
  - [src/utils/CLAUDE.md](src/utils/CLAUDE.md)
