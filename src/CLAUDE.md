# FinancialQA Source Code Structure

**Last Updated:** 2026-01-11
**Research Status:** Complete Architecture Analysis

## Overview

The FinancialQA project implements RAG (Retrieval-Augmented Generation) experiments for financial question answering using the FinDER dataset, with Langfuse integration for experiment tracking and evaluation.

**Critical Architecture Issue:** The codebase has two separate evaluation systems with the new Langfuse-based system using placeholder implementations instead of the actual RAG pipeline.

## Key Components

### 1. Langfuse Integration (`langfuse_integration/`)

**Purpose:** Experiment tracking, evaluation, and dataset management via Langfuse SDK

**Status:** ⚠️ Partially implemented - evaluators work but RAG integration is placeholder

**Essential Files:**
- `config.py` - Langfuse credential validation
- `models.py` - Pydantic models for experiments (ExperimentRunOutput, ItemExecutionResult, etc.)
- `evaluators.py` - Item-level and run-level evaluation functions
- `experiment_runner.py` - Main orchestration (NEEDS FIX - uses placeholders)
- `exceptions.py` - Custom exception types
- `dataset_manager.py` - Upload FinDER to Langfuse (one-time utility)

**Non-Essential / Over-Engineered:**
- `analysis.py` - Experiment comparison (not used by main workflow)
- `client.py` - Wrapper around Langfuse SDK (redundant)
- `retry.py` - Retry logic (minimal usage)

**Critical Bug:**
```python
# In experiment_runner.py line 90-104
def baseline_task(*, item, **kwargs):
    # TODO: Replace with actual RAG pipeline implementation
    logger.warning("Using placeholder RAG implementation")
    result = {
        "answer": f"Placeholder answer for: {query_text}",
        "retrieved_passages": [],  # ALWAYS EMPTY!
    }
    return result
```

**Why This Matters:**
- Experiments run with mock data, not actual RAG
- Evaluators score placeholder responses, not real answers
- Retrieval implementations exist but aren't called

**Fix Required:**
Replace placeholder with actual BaselineRAG instantiation:
```python
from src.rag.baseline import BaselineRAG
pipeline = BaselineRAG.from_config(config)
result = pipeline.process_query(query)
```

### 2. RAG Pipeline (`rag/`)

**Purpose:** RAG pipeline implementations

**Files:**
- `baseline.py` - Single-agent RAG (retrieve → generate)

**Status:** ✅ Fully implemented BUT not used by Langfuse experiments

**Usage:**
- Used by OLD evaluation system (`src/evaluation/runner.py`)
- NOT used by NEW Langfuse system (`scripts/run_experiment.py`)

**Integration Point:**
Should be instantiated in `langfuse_integration/experiment_runner.py` task functions but currently isn't.

### 3. Retrieval (`retrieval/`)

**Purpose:** Passage retrieval strategies

**Implementations:**
- `base.py` - Abstract RetrieverBase class
- `bm25.py` - Keyword-based retrieval (BM25Okapi)
- `dense.py` - Semantic retrieval (sentence-transformers + ChromaDB)
- `hybrid.py` - Combined BM25 + dense with rank fusion

**Status:** ✅ All fully implemented BUT not used in Langfuse experiments

**Dependencies:**
- BM25: `rank-bm25` library
- Dense: `sentence-transformers`, `chromadb`
- Hybrid: Both of above

**Integration Status:**
- Used by BaselineRAG pipeline ✅
- BaselineRAG used by old evaluation system ✅
- BaselineRAG NOT used by Langfuse experiments ❌

**Decision Needed:**
If only BM25 is needed, can delete `dense.py` and `hybrid.py` to reduce dependencies.

### 4. Data Handler (`data_handler/`)

**Purpose:** Dataset loading and preprocessing

**Files:**
- `models.py` - Data structures (Query, EvidencePassage, RetrievalResult)
- `loader.py` - FinDER dataset loader from HuggingFace
- `indexer.py` - Build/save retrieval indexes
- `preprocessor.py` - Query enrichment and validation

**Usage Analysis:**

| File | Used By | Status |
|------|---------|--------|
| models.py | Many files | ✅ Essential (type definitions) |
| loader.py | upload_dataset.py | ⚠️ Used for one-time upload only |
| indexer.py | Old evaluation system | ❌ Not used in Langfuse workflow |
| preprocessor.py | Nothing | ❌ Completely unused - DELETE |

**Preprocessor (315 lines) - DELETE:**
- Never imported anywhere
- Complex query enrichment (domain terms, ambiguity, subdomain classification)
- Over-engineered for unused features

**Loader - SIMPLIFY:**
- Only needed for `scripts/upload_dataset.py` (one-time)
- Could be simplified to just load queries without all metadata enrichment

**Indexer - OPTIONAL:**
- Used only by old evaluation system
- Retrievers can build indexes on-demand
- Can delete if consolidating on Langfuse workflow

### 5. Agents (`agents/`)

**Purpose:** Multi-agent RAG system

**Status:** ❌ **SCAFFOLDING ONLY - NO IMPLEMENTATIONS**

**Files:**
- `base.py` - Abstract Agent class, AgentConfig, AgentExecution

**Reality Check:**
- Only base classes exist
- No concrete agent implementations (QueryUnderstanding, ContextResolution, etc.)
- Config supports multi-agent but task function has placeholder
- No multi-agent configs in `experiments/configs/`

**Verdict:** DELETE entire directory unless multi-agent is planned soon

### 6. Evaluation (`evaluation/`)

**Purpose:** OLD evaluation system (pre-Langfuse)

**Files:**
- `runner.py` - EvaluationRunner orchestration
- `metrics.py` - Metric calculators
- `logger.py` - Langfuse logging (OLD approach)
- `reporter.py` - Report generation
- `models.py` - Evaluation data structures

**Status:** ⚠️ **SEPARATE WORKFLOW FROM LANGFUSE EXPERIMENTS**

**Key Difference:**

| Feature | Old System (evaluation/) | New System (langfuse_integration/) |
|---------|-------------------------|-----------------------------------|
| Entry Point | `src.evaluation.runner` | `scripts/run_experiment.py` |
| RAG Integration | ✅ Uses actual BaselineRAG | ❌ Uses placeholders |
| Retrieval | ✅ Actually retrieves | ❌ Returns empty list |
| Langfuse | Basic logging | Full SDK integration |
| Dataset | Loads via FinDERLoader | Loads from Langfuse dataset |

**Decision Needed:**
1. Keep old system for local testing? OR
2. Delete and consolidate on Langfuse workflow (after fixing placeholders)?

**Recommendation:** Delete after integrating BaselineRAG into Langfuse workflow

### 7. Configuration (`config/`)

**Purpose:** Experiment configuration management

**Files:**
- `experiments.py` - ExperimentConfig and LangfuseExperimentConfig

**Config Hierarchy:**
```
ExperimentConfig (base)
  - Used by old evaluation system
  - Fields: name, description, retrieval_config, llm_configs, etc.

LangfuseExperimentConfig (extends base)
  - Adds Langfuse-specific settings
  - Additional fields: dataset_name, evaluators, concurrency, etc.
```

**Over-Engineering:**
- `agent_architecture` field - No multi-agent implementation
- `orchestration_mode` field - No multi-agent implementation
- `use_local_data` + `local_data_path` - Not implemented (line 236 TODO)

**Simplification:**
Remove unused fields or mark as Optional[...] = None for future use

### 8. Utilities (`utils/`)

**Purpose:** Shared utilities

**Files:**
- `llm_client.py` - LLM provider abstraction (OpenAI, Anthropic)
- `cache.py` - Result caching (if exists)

**Status:** Used by BaselineRAG for answer generation

## Architecture Patterns

### 1. Config-Driven Design

**Intended:**
```yaml
retrieval_config:
  strategy: "bm25"  # Could be "dense" or "hybrid"
  top_k: 5
```
→ Code should instantiate BM25Retriever, DenseRetriever, or HybridRetriever

**Reality:**
Config is read but ignored; placeholder returns empty retrieved_passages

**Fix Needed:**
Make `_create_baseline_task()` actually use config to build pipeline

### 2. Dual Evaluation Systems (Problematic)

**Problem:** Two completely separate code paths:

1. **Old System:** Makefile → `src.evaluation.runner` → BaselineRAG → Actual retrieval
2. **New System:** Makefile → `scripts/run_experiment.py` → Placeholders → Mock data

**Why This Exists:**
Langfuse integration was added as new system but RAG integration incomplete

**Resolution:**
Merge approaches - use Langfuse workflow with actual RAG pipeline

### 3. Placeholder Pattern (Anti-Pattern)

Multiple TODOs with placeholder implementations:
- Line 90: "TODO: Replace with actual RAG pipeline implementation"
- Line 152: "TODO: Replace with actual multi-agent RAG implementation"
- Line 236: "TODO: Implement local data loading"

**This indicates:**
- Feature was planned but not completed
- Evaluation framework built before RAG integration
- Need to either implement or remove scaffolding

## Dependencies

### Production
- `langfuse` - Experiment tracking SDK
- `pydantic` - Data validation
- `rank-bm25` - BM25 retrieval
- `sentence-transformers` - Dense embeddings (optional if not using dense retrieval)
- `chromadb` - Vector database (optional if not using dense retrieval)
- `datasets` - HuggingFace datasets for FinDER
- OpenAI/Anthropic SDKs - LLM providers

### Development
- `pytest` - Testing
- `black`, `flake8`, `mypy` - Code quality

## Integration Points

### Dataset Upload Flow
```
scripts/upload_dataset.py
  → FinDERLoader.load()
  → upload_finder_dataset()
  → Langfuse.create_dataset_item()
```

### Experiment Execution Flow (Current - BROKEN)
```
scripts/run_experiment.py
  → LangfuseExperimentConfig.from_yaml()
  → run_langfuse_experiment()
    → create_task_function()  [Returns PLACEHOLDER function]
    → register_evaluators()
    → dataset.run_experiment()  [Evaluates PLACEHOLDER results]
```

### Experiment Execution Flow (Should Be)
```
scripts/run_experiment.py
  → LangfuseExperimentConfig.from_yaml()
  → run_langfuse_experiment()
    → create_task_function()
      → BaselineRAG.from_config()  [Actual RAG instance]
      → retriever.retrieve()  [Actual retrieval]
      → llm_client.generate()  [Actual generation]
    → register_evaluators()
    → dataset.run_experiment()  [Evaluates REAL results]
```

## Common Issues

### 1. "Experiments return placeholder answers"
**Cause:** Task function in experiment_runner.py uses hardcoded placeholder
**Fix:** Integrate BaselineRAG.process_query()

### 2. "Retrieved passages always empty"
**Cause:** Placeholder returns `"retrieved_passages": []`
**Fix:** Call retriever.retrieve() and return actual passages

### 3. "Retrieval implementations not used"
**Cause:** Task function doesn't instantiate retrievers
**Fix:** Use config.retrieval_config to create retriever in task function

### 4. "Multi-agent not working"
**Cause:** No concrete agent implementations exist
**Fix:** Either implement agents or remove multi-agent config options

### 5. "Old evaluation vs. new evaluation confusion"
**Cause:** Two separate systems coexist
**Fix:** Consolidate on one system (recommend Langfuse after fixing)

## Best Practices

### When Adding New Features

1. **Check if placeholder** - Search for TODO comments
2. **Verify integration** - Don't just add config, actually call the code
3. **Test end-to-end** - Run actual experiment, not just unit tests
4. **Remove unused** - Delete scaffolding if feature isn't being built

### When Running Experiments

1. **Use Langfuse workflow** - `scripts/run_experiment.py`
2. **Verify actual RAG** - Check that answers aren't "Placeholder answer for: ..."
3. **Monitor retrieved passages** - Should not be empty if retrieval working

### When Modifying Configs

1. **Check if field is used** - Many fields in LangfuseExperimentConfig unused
2. **Validate strategy** - If changing retrieval strategy, verify it's implemented
3. **Test with subset** - Use `--max-items 10` for quick iteration

## Recommendations

### Immediate (Fix Broken Functionality)
1. ✅ Integrate BaselineRAG into experiment_runner.py task functions
2. ✅ Remove placeholder implementations
3. ✅ Test that retrieval actually returns passages

### Short-Term (Clean Up)
1. ❌ Delete `src/data_handler/preprocessor.py` (unused)
2. ❌ Delete `src/agents/` (no implementations)
3. ❌ Delete `src/langfuse_integration/analysis.py` (not in workflow)
4. ❌ Delete `src/langfuse_integration/client.py` (redundant)
5. ⚠️ Decide: Keep or delete `src/evaluation/` (old system)

### Medium-Term (Simplify)
1. Remove unused config fields (agent_architecture, local_data_path)
2. Consolidate on single evaluation system
3. Consider deleting dense/hybrid retrievers if only using BM25

### Long-Term (Optimize)
1. Make config truly drive execution (instantiate retrievers from config)
2. Add actual multi-agent if needed, or remove scaffolding
3. Reduce dependencies (remove ChromaDB if not using dense retrieval)

## Questions to Resolve

1. **Old Evaluation System:** Delete `src/evaluation/` or keep for non-Langfuse usage?
2. **Multi-Agent:** Implement or remove all agent-related code?
3. **Dense/Hybrid Retrieval:** Keep or delete if only using BM25?
4. **Local Data Loading:** Implement or remove use_local_data config?
5. **Dataset Preprocessor:** Why created if never used?

---

**For New Contributors:**
Start by reading `RESEARCH_REPORT.md` for detailed analysis, then review this file for component overview. The key issue to understand is that the Langfuse experiment workflow is incomplete - it uses placeholder task functions instead of the actual RAG pipeline that exists in `src/rag/baseline.py`.
