# Retrieval Module

## Purpose

Pluggable retrieval implementations for finding relevant evidence passages. Provides a common interface (`RetrieverBase`) with multiple strategy implementations (BM25, dense, hybrid).

## Key Components

### 1. Base Interface ([base.py](base.py))

**`RetrieverBase`** - Abstract interface for all retrievers

```python
class RetrieverBase(ABC):
    @abstractmethod
    def index_corpus(self, passages: List[EvidencePassage]) -> None:
        """One-time indexing of evidence passages"""

    @abstractmethod
    def retrieve(self, query_text: str, top_k: int = 5) -> RetrievalResult:
        """Return ranked passages for query"""
```

**Exceptions:**
- `RetrievalError` - Base exception for retrieval failures
- `IndexNotBuiltError` - Raised if retrieve() called before index_corpus()

### 2. BM25 Retrieval ([bm25.py](bm25.py)) - **PRIMARY**

**`BM25Retriever`** - Keyword-based retrieval using rank_bm25

#### Configuration

```python
BM25Retriever(config: Dict[str, Any])
# Config parameters:
# - k1: float = 1.5 (term frequency saturation)
# - b: float = 0.75 (length normalization)
# - top_k: int = 5 (default number of results)
```

#### Implementation Details

**Indexing:**
```python
def index_corpus(self, passages: List[EvidencePassage]):
    # Tokenize each passage (whitespace split, lowercase)
    tokenized_corpus = [p.text.lower().split() for p in passages]

    # Build BM25 index with parameters
    self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
    self.passages = passages
```

**Retrieval:**
```python
def retrieve(self, query_text: str, top_k: int = 5) -> RetrievalResult:
    # Tokenize query same way
    tokenized_query = query_text.lower().split()

    # Get BM25 scores (higher = more relevant)
    scores = self.bm25.get_scores(tokenized_query)

    # Get top-K indices
    top_indices = np.argsort(scores)[::-1][:top_k]

    # Create RetrievedPassage objects (with 1-indexed rank)
    retrieved_passages = [
        RetrievedPassage(
            passage=self.passages[idx],
            score=float(scores[idx]),
            rank=i + 1
        )
        for i, idx in enumerate(top_indices)
    ]

    return RetrievalResult(
        query_id="",
        retrieved_passages=retrieved_passages,
        strategy="bm25",
        retrieval_time_seconds=elapsed,
        metadata={"k1": self.k1, "b": self.b, "top_k": top_k}
    )
```

#### BM25 Parameters

- **k1** (term frequency saturation): Controls how quickly the impact of term frequency saturates
  - Lower (0.5-1.2): More aggressive saturation (good for verbose documents)
  - Default (1.5): Balanced
  - Higher (2.0-3.0): Less saturation (good for short documents)

- **b** (length normalization): Controls how much document length affects scoring
  - 0: No length normalization
  - Default (0.75): Moderate normalization
  - 1: Full normalization (penalizes long documents heavily)

#### Status

✅ **ACTIVELY USED** - Primary retrieval strategy for baseline experiments

### 3. Dense Retrieval ([dense.py](dense.py)) - **OPTIONAL**

**`DenseRetriever`** - Semantic search using sentence embeddings

#### Configuration

```python
DenseRetriever(config: Dict[str, Any])
# Config parameters:
# - embedding_model: str = "all-MiniLM-L6-v2"
# - collection_name: str = "financial_qa"
# - similarity_metric: str = "cosine"
# - top_k: int = 5
```

#### Implementation Details

**Dependencies:**
- `sentence-transformers` - Embedding model
- `chromadb` - Vector database

**Indexing:**
```python
def index_corpus(self, passages: List[EvidencePassage]):
    # Initialize embedding model
    self.embedder = SentenceTransformer(self.embedding_model)

    # Create ChromaDB collection
    self.collection = self.chroma_client.create_collection(
        name=self.collection_name,
        metadata={"hnsw:space": self.similarity_metric}
    )

    # Embed passages in batches
    texts = [p.text for p in passages]
    embeddings = self.embedder.encode(texts, batch_size=32)

    # Add to ChromaDB
    self.collection.add(
        ids=[p.id for p in passages],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[p.metadata.dict() for p in passages]
    )
```

**Retrieval:**
```python
def retrieve(self, query_text: str, top_k: int = 5) -> RetrievalResult:
    # Embed query
    query_embedding = self.embedder.encode([query_text])[0]

    # Query ChromaDB
    results = self.collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    # Convert to RetrievedPassage objects
    # ...
```

#### Status

⚠️ **NOT ACTIVELY TESTED** - Available but not in default workflow

Dependencies are optional. To use:
```bash
pip install sentence-transformers chromadb
```

### 4. Hybrid Retrieval ([hybrid.py](hybrid.py)) - **OPTIONAL**

**`HybridRetriever`** - Combined BM25 + dense with rank fusion

#### Configuration

```python
HybridRetriever(config: Dict[str, Any])
# Config parameters:
# - bm25_config: Dict (passed to BM25Retriever)
# - dense_config: Dict (passed to DenseRetriever)
# - bm25_weight: float = 0.5
# - dense_weight: float = 0.5
# - fusion_method: str = "rrf" (reciprocal rank fusion) or "score"
# - top_k: int = 5
```

#### Implementation Details

**Indexing:**
```python
def index_corpus(self, passages: List[EvidencePassage]):
    # Index with both retrievers
    self.bm25_retriever.index_corpus(passages)
    self.dense_retriever.index_corpus(passages)
```

**Retrieval:**
```python
def retrieve(self, query_text: str, top_k: int = 5) -> RetrievalResult:
    # Retrieve from both
    bm25_result = self.bm25_retriever.retrieve(query_text, top_k * 2)
    dense_result = self.dense_retriever.retrieve(query_text, top_k * 2)

    # Fuse results
    if self.fusion_method == "rrf":
        fused = self._reciprocal_rank_fusion(bm25_result, dense_result)
    else:
        fused = self._score_fusion(bm25_result, dense_result)

    # Return top-K
    return RetrievalResult(
        retrieved_passages=fused[:top_k],
        strategy="hybrid",
        ...
    )
```

**Reciprocal Rank Fusion:**
```python
def _reciprocal_rank_fusion(self, result1, result2, k=60):
    # RRF score = sum(1 / (k + rank_in_source))
    # Combines rankings from multiple sources
    # k=60 is standard parameter
```

**Score Fusion:**
```python
def _score_fusion(self, result1, result2):
    # Weighted average of normalized scores
    # score = bm25_weight * norm(bm25_score) + dense_weight * norm(dense_score)
```

#### Status

⚠️ **NOT ACTIVELY TESTED** - Proof-of-concept implementation

Requires both BM25 and dense dependencies.

---

## Usage Examples

### BM25 Retrieval (Standard Workflow)

```python
from src.retrieval.bm25 import BM25Retriever
from src.data_handler.loader import FinDERLoader

# Load corpus
loader = FinDERLoader()
corpus = loader.load_corpus()

# Create and index retriever
retriever = BM25Retriever({"k1": 1.5, "b": 0.75, "top_k": 5})
retriever.index_corpus(corpus)

# Retrieve
query_text = "What is the difference between a forward and futures contract?"
result = retriever.retrieve(query_text, top_k=5)

# Access results
for retrieved in result.retrieved_passages:
    print(f"Rank {retrieved.rank}: {retrieved.passage.text[:100]}...")
    print(f"Score: {retrieved.score}")
```

### Dense Retrieval (Optional)

```python
from src.retrieval.dense import DenseRetriever

retriever = DenseRetriever({
    "embedding_model": "all-MiniLM-L6-v2",
    "collection_name": "financial_qa",
    "similarity_metric": "cosine"
})

retriever.index_corpus(corpus)
result = retriever.retrieve(query_text, top_k=5)
```

### Hybrid Retrieval (Optional)

```python
from src.retrieval.hybrid import HybridRetriever

retriever = HybridRetriever({
    "bm25_config": {"k1": 1.5, "b": 0.75},
    "dense_config": {"embedding_model": "all-MiniLM-L6-v2"},
    "bm25_weight": 0.6,
    "dense_weight": 0.4,
    "fusion_method": "rrf"
})

retriever.index_corpus(corpus)
result = retriever.retrieve(query_text, top_k=5)
```

---

## Configuration in Experiments

Retrieval strategy is specified in experiment YAML config:

```yaml
# BM25 (default)
retrieval_config:
  strategy: "bm25"
  top_k: 5
  k1: 1.5
  b: 0.75

# Dense (optional)
retrieval_config:
  strategy: "dense"
  top_k: 5
  embedding_model: "all-MiniLM-L6-v2"
  similarity_metric: "cosine"

# Hybrid (optional)
retrieval_config:
  strategy: "hybrid"
  top_k: 5
  bm25_weight: 0.6
  dense_weight: 0.4
  fusion_method: "rrf"
```

The experiment runner (`src/langfuse_integration/experiment_runner.py`) instantiates the appropriate retriever based on `strategy`.

---

## Performance Characteristics

| Strategy | Speed | Semantic Understanding | Dependencies | Status |
|----------|-------|----------------------|--------------|--------|
| **BM25** | Fast | Keyword-based | rank_bm25 (lightweight) | ✅ Active |
| **Dense** | Slower | Semantic | sentence-transformers, chromadb | ⚠️ Optional |
| **Hybrid** | Slowest | Best of both | Both above | ⚠️ Optional |

### BM25 Strengths
- Fast indexing and retrieval
- Works well for exact term matches
- No GPU required
- Lightweight dependencies

### BM25 Weaknesses
- No semantic understanding
- Struggles with synonyms/paraphrasing
- Sensitive to vocabulary mismatch

### Dense Strengths
- Semantic understanding
- Handles synonyms/paraphrasing
- Robust to vocabulary mismatch

### Dense Weaknesses
- Slower than BM25
- Requires more memory
- May miss exact term matches
- Requires GPU for fast encoding (optional)

### Hybrid Strengths
- Combines keyword and semantic signals
- Most robust retrieval

### Hybrid Weaknesses
- Slowest (runs both retrievers)
- Most complex to tune (multiple hyperparameters)

---

## Current Status Summary

| Component | Status | Usage |
|-----------|--------|-------|
| BM25Retriever | ✅ Active | Primary retrieval strategy |
| DenseRetriever | ⚠️ Available | Exists but not tested in workflow |
| HybridRetriever | ⚠️ Available | Exists but not tested in workflow |
| RetrieverBase | ✅ Active | Interface for all retrievers |

---

## Dependencies

### Required (BM25)
- `rank-bm25` - BM25 implementation
- `numpy` - Array operations

### Optional (Dense)
- `sentence-transformers` - Embedding models
- `chromadb` - Vector database

### Optional (Hybrid)
- Both BM25 and dense dependencies

---

## Extension: Adding New Retrieval Strategy

```python
# 1. Create new retriever in src/retrieval/custom.py
from src.retrieval.base import RetrieverBase
from src.data_handler.models import EvidencePassage, RetrievalResult

class CustomRetriever(RetrieverBase):
    def __init__(self, config: Dict[str, Any]):
        # Initialize with config
        pass

    def index_corpus(self, passages: List[EvidencePassage]) -> None:
        # Build your index
        pass

    def retrieve(self, query_text: str, top_k: int = 5) -> RetrievalResult:
        # Retrieve and rank passages
        pass

# 2. Add to config validation in src/config/experiments.py
valid_strategies = {"bm25", "dense", "hybrid", "custom"}

# 3. Update experiment_runner.py to instantiate
elif strategy == "custom":
    from src.retrieval.custom import CustomRetriever
    retriever = CustomRetriever(retriever_config)
```

---

## See Also

- Main Architecture: [../../ARCHITECTURE.md](../../ARCHITECTURE.md)
- Data Handler: [../data_handler/CLAUDE.md](../data_handler/CLAUDE.md)
- RAG Pipeline: [../rag/CLAUDE.md](../rag/CLAUDE.md)
- Experiment Runner: [../langfuse_integration/CLAUDE.md](../langfuse_integration/CLAUDE.md)
