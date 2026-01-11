# Data Handler Module

## Purpose

Manage dataset loading, transformation, and core data structures for queries and evidence passages. This module provides the foundation for working with the FinDER financial QA dataset.

## Key Components

### 1. Data Models ([models.py](models.py))

#### Query Models

**`Query`** - Financial question representation
```python
Query(
    id: str,                          # Unique query identifier
    text: str,                        # Question text
    expected_answer: str,             # Ground truth answer
    expected_evidence: List[str],     # List of relevant passage IDs
    metadata: QueryMetadata           # Complexity and characteristics
)
```

**`QueryMetadata`** - Query complexity and characteristics
```python
QueryMetadata(
    domain_term_count: int,           # Number of financial terms
    has_ambiguity: bool,              # Contains ambiguous abbreviations
    query_type: str,                  # "definition", "comparison", "calculation", etc.
    required_evidence_count: int,     # Number of relevant passages
    financial_subdomain: str,         # "equity", "bonds", "derivatives", etc.
    reasoning_required: bool          # Multi-step reasoning needed
)
```

#### Passage Models

**`EvidencePassage`** - Text passage from corpus
```python
EvidencePassage(
    id: str,                          # Unique passage identifier
    text: str,                        # Passage content
    document_id: str,                 # Source document ID
    metadata: EvidenceMetadata        # Source information
)
```

**`EvidenceMetadata`** - Passage source information
```python
EvidenceMetadata(
    source_type: str,                 # "textbook", "regulation", "report", etc.
    publication_date: Optional[str],  # When published
    page_number: Optional[int],       # Page in source document
    section: Optional[str]            # Section/chapter name
)
```

#### Retrieval Result Models

**`RetrievedPassage`** - Ranked passage with relevance score
```python
RetrievedPassage(
    passage: EvidencePassage,         # The actual passage
    score: float,                     # Relevance score
    rank: int,                        # 1-indexed rank in results
    is_relevant: Optional[bool]       # True if in expected_evidence
)
```

**`RetrievalResult`** - Complete retrieval output
```python
RetrievalResult(
    query_id: str,                    # Query this result is for
    retrieved_passages: List[RetrievedPassage],
    strategy: str,                    # "bm25", "dense", "hybrid"
    retrieval_time_seconds: float,    # Time taken to retrieve
    metadata: Dict[str, Any]          # Additional info
)
```

### 2. Dataset Loader ([loader.py](loader.py))

**`FinDERLoader`** - HuggingFace dataset loader for FinDER

#### Main Methods

**`download(cache_dir="./data/finder")`**
- Downloads dataset from `"Linq-AI-Research/FinDER"` (HuggingFace)
- Caches locally to avoid repeated downloads
- Returns: HuggingFace `DatasetDict`

**`load(cache_dir="./data/finder", split="train")`**
- Converts dataset to `List[Query]` with metadata enrichment
- Automatically classifies query complexity
- Detects financial terminology and ambiguity
- Returns: `List[Query]`

**`load_corpus(cache_dir="./data/finder")`**
- Extracts unique evidence passages from all query references
- Deduplicates by passage ID
- Returns: `List[EvidencePassage]`

**`get_dataset_version(cache_dir="./data/finder")`**
- Reads version metadata from dataset
- Returns: Version string (e.g., "1.0.0")

#### Helper Methods

**`_parse_example(example_dict)`**
- Parses single dataset item into `Query` object
- Enriches with metadata (domain terms, ambiguity, query type)

**`_count_domain_terms(text)`**
- Counts financial terminology in query
- Uses predefined financial term list
- Returns: Count of domain-specific terms

**`_detect_ambiguity(text)`**
- Identifies ambiguous abbreviations (MS, GS, PE, etc.)
- These can refer to multiple entities (Microsoft/Morgan Stanley, etc.)
- Returns: `True` if ambiguous terms found

**`_classify_query_type(text)`**
- Categorizes query into types
- Types: "definition", "comparison", "calculation", "explanation", "fact_retrieval"
- Uses keyword patterns
- Returns: Query type string

#### Usage Example

```python
from src.data_handler.loader import FinDERLoader

loader = FinDERLoader()

# Load all queries with metadata
queries = loader.load()
print(f"Loaded {len(queries)} queries")

# Load corpus for retrieval indexing
corpus = loader.load_corpus()
print(f"Corpus has {len(corpus)} passages")

# Check version
version = loader.get_dataset_version()
print(f"Dataset version: {version}")
```

### 3. Index Builder ([indexer.py](indexer.py))

**`IndexBuilder`** - Persistent index utilities (optional)

#### Methods

**`build_bm25_index(corpus, index_path)`**
- Builds BM25 index from corpus
- Serializes to disk for later loading
- Returns: BM25 index object

**`load_bm25_index(index_path, corpus)`**
- Loads previously built BM25 index
- Requires corpus for passage lookup
- Returns: BM25 index object

**Note:** NOT used in current Langfuse workflow. Indexes are built on-demand in experiment runner for simplicity.

---

## Data Flow

### Query Loading Flow

```
HuggingFace Dataset
    ↓ (download)
Local Cache (./data/finder)
    ↓ (load)
Parse Examples
    ↓ (_parse_example)
Enrich Metadata
    ├─ Count domain terms
    ├─ Detect ambiguity
    └─ Classify query type
    ↓
List[Query] with full metadata
```

### Corpus Loading Flow

```
HuggingFace Dataset
    ↓
Extract all evidence references
    ↓
Deduplicate by passage ID
    ↓
Parse passage metadata
    ↓
List[EvidencePassage]
```

---

## Data Model Relationships

```
Query
  ├─ text (question)
  ├─ expected_answer (ground truth)
  ├─ expected_evidence (List[passage_id])
  └─ metadata (QueryMetadata)

EvidencePassage
  ├─ id (matches IDs in Query.expected_evidence)
  ├─ text (passage content)
  └─ metadata (EvidenceMetadata)

RetrievalResult
  ├─ query_id (matches Query.id)
  └─ retrieved_passages (List[RetrievedPassage])
      └─ passage (EvidencePassage)
          └─ can be matched against Query.expected_evidence
```

---

## Current Status

✅ **Actively Used** - Core data infrastructure

Used by:
- [src/langfuse_integration/experiment_runner.py](../langfuse_integration/experiment_runner.py) - Loads corpus for indexing
- [scripts/upload_dataset.py](../../scripts/upload_dataset.py) - Loads queries for Langfuse upload
- [src/rag/baseline.py](../rag/baseline.py) - Uses Query and RetrievalResult models
- [src/retrieval/*.py](../retrieval/) - Uses EvidencePassage and RetrievalResult models

---

## Dependencies

### Internal
- None (foundational module)

### External
- `datasets` - HuggingFace datasets library (required)
- `pydantic` or `dataclasses` - Data validation (depending on implementation)

---

## Configuration

Dataset caching location can be configured:

```python
loader = FinDERLoader()
queries = loader.load(cache_dir="./custom/cache/path")
corpus = loader.load_corpus(cache_dir="./custom/cache/path")
```

Default: `./data/finder` (gitignored)

---

## Example Data

### Sample Query

```python
Query(
    id="finder_001",
    text="What is the difference between a forward contract and a futures contract?",
    expected_answer="A forward contract is a customized agreement...",
    expected_evidence=["passage_123", "passage_456"],
    metadata=QueryMetadata(
        domain_term_count=4,
        has_ambiguity=False,
        query_type="comparison",
        required_evidence_count=2,
        financial_subdomain="derivatives",
        reasoning_required=True
    )
)
```

### Sample Evidence Passage

```python
EvidencePassage(
    id="passage_123",
    text="Forward contracts are bilateral agreements to buy or sell...",
    document_id="derivatives_handbook_ch3",
    metadata=EvidenceMetadata(
        source_type="textbook",
        publication_date="2023-01",
        page_number=45,
        section="Chapter 3: Derivatives Basics"
    )
)
```

### Sample Retrieval Result

```python
RetrievalResult(
    query_id="finder_001",
    retrieved_passages=[
        RetrievedPassage(
            passage=EvidencePassage(id="passage_123", ...),
            score=8.45,
            rank=1,
            is_relevant=True
        ),
        RetrievedPassage(
            passage=EvidencePassage(id="passage_789", ...),
            score=7.32,
            rank=2,
            is_relevant=False
        ),
    ],
    strategy="bm25",
    retrieval_time_seconds=0.032,
    metadata={"top_k": 5, "k1": 1.5, "b": 0.75}
)
```

---

## Metadata Enrichment Details

### Domain Term Detection

Financial terms recognized:
- Derivatives: "forward", "futures", "option", "swap", "derivative"
- Equity: "stock", "equity", "dividend", "share", "market cap"
- Fixed Income: "bond", "yield", "coupon", "maturity", "duration"
- General: "portfolio", "risk", "return", "valuation", "hedging"

### Ambiguity Detection

Ambiguous abbreviations:
- MS → Microsoft or Morgan Stanley
- GS → Goldman Sachs or General Services
- PE → Private Equity or Price-to-Earnings
- VC → Venture Capital or Vanguard Corporation
- BA → Bank of America or Business Analysis

### Query Type Classification

| Type | Keywords | Example |
|------|----------|---------|
| definition | "what is", "define", "meaning of" | "What is a call option?" |
| comparison | "difference between", "compare", "versus" | "Stocks vs bonds?" |
| calculation | "calculate", "compute", "how much" | "How to calculate NPV?" |
| explanation | "why", "how does", "explain" | "Why do bond prices fall?" |
| fact_retrieval | "when", "who", "where" | "When was the SEC founded?" |

---

## See Also

- Main Architecture: [../../ARCHITECTURE.md](../../ARCHITECTURE.md)
- Retrieval Module: [../retrieval/CLAUDE.md](../retrieval/CLAUDE.md)
- RAG Pipeline: [../rag/CLAUDE.md](../rag/CLAUDE.md)
- Langfuse Integration: [../langfuse_integration/CLAUDE.md](../langfuse_integration/CLAUDE.md)
