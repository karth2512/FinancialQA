"""
GraphRAG knowledge graph-based retrieval using Python library.

This module implements GraphRAGRetriever which uses Microsoft GraphRAG's
knowledge graph approach for entity-aware retrieval. Unlike the previous
CLI-based implementation, this uses the Python library directly with
custom Anthropic ChatModel and HuggingFace EmbeddingModel.

No LiteLLM proxy required.
"""

import time
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

from src.retrieval.base import RetrieverBase, RetrievalError, IndexNotBuiltError
from src.data_handler.models import EvidencePassage, RetrievalResult, RetrievedPassage

logger = logging.getLogger(__name__)


class GraphRAGRetriever(RetrieverBase):
    """
    GraphRAG-based retrieval using Python library (not CLI).

    This retriever loads a pre-built GraphRAG index and supports two query modes:
    - local: Entity-focused retrieval (specific entities and neighborhoods)
    - global: Community-level retrieval (high-level summaries)

    Uses custom Anthropic ChatModel and HuggingFace EmbeddingModel
    registered via src.graphrag_models, eliminating the need for LiteLLM proxy.

    Note: Requires pre-built GraphRAG index in the specified output directory.
    Build index with: python scripts/graphrag_index_python.py
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GraphRAG retriever.

        Args:
            config: Configuration dict with keys:
                - graphrag_root: Path to GraphRAG directory (default: "./data/graphrag")
                - method: Query method "local" or "global" (default: "local")
                - top_k: Number of results to return (default: 5)
                - community_level: Community level for global search (default: 2)
                - response_type: Response type (default: "Multiple Paragraphs")
                - chat_model: Anthropic model name (default: "claude-3-5-haiku-20241022")
                - embedding_model: HuggingFace model name (default: "all-MiniLM-L6-v2")
        """
        super().__init__(config)

        self.graphrag_root = Path(config.get("graphrag_root", "./data/graphrag"))
        self.method = config.get("method", "local").lower()
        self.top_k = config.get("top_k", 5)
        self.community_level = config.get("community_level", 2)
        self.response_type = config.get("response_type", "Multiple Paragraphs")

        # Model configuration
        self.chat_model_name = config.get("chat_model", "claude-3-5-haiku-20241022")
        self.embedding_model_name = config.get("embedding_model", "all-MiniLM-L6-v2")

        # Validate method
        if self.method not in ["local", "global"]:
            raise ValueError(
                f"Invalid method: {self.method}. Must be 'local' or 'global'"
            )

        # Paths
        self.output_dir = self.graphrag_root / "output"

        # Loaded data (initialized in index_corpus)
        self.entities: Optional[pd.DataFrame] = None
        self.relationships: Optional[pd.DataFrame] = None
        self.communities: Optional[pd.DataFrame] = None
        self.community_reports: Optional[pd.DataFrame] = None
        self.text_units: Optional[pd.DataFrame] = None

        # Models (initialized in index_corpus)
        self.chat_model: Optional[Any] = None
        self.embedding_model: Optional[Any] = None

        # Entity name to ID mapping for fast lookup
        self._entity_name_to_id: Dict[str, str] = {}
        self._entity_id_to_data: Dict[str, Dict] = {}

        logger.info(
            f"Initialized GraphRAGRetriever: method={self.method}, "
            f"chat_model={self.chat_model_name}, embedding_model={self.embedding_model_name}"
        )

    def index_corpus(self, passages: List[EvidencePassage]) -> None:
        """
        Load pre-built GraphRAG index from parquet files.

        This method doesn't build a new index, but loads an existing one
        created by scripts/graphrag_index_python.py.

        Args:
            passages: Evidence passages (not used - index already built)

        Raises:
            RetrievalError: If GraphRAG index not found or loading fails
        """
        try:
            # Check if output directory exists
            if not self.output_dir.exists():
                raise RetrievalError(
                    "graphrag",
                    f"GraphRAG output directory not found at {self.output_dir}. "
                    "Run 'python scripts/graphrag_index_python.py' first to build the index.",
                    "",
                )

            # Check for required parquet files
            required_files = [
                "create_final_entities.parquet",
                "create_final_relationships.parquet",
                "create_final_communities.parquet",
                "create_final_community_reports.parquet",
                "create_final_text_units.parquet",
            ]

            missing_files = []
            for filename in required_files:
                file_path = self.output_dir / filename
                if not file_path.exists():
                    missing_files.append(filename)

            if missing_files:
                raise RetrievalError(
                    "graphrag",
                    f"Missing required GraphRAG files: {', '.join(missing_files)}. "
                    "Index may be incomplete. Re-run 'python scripts/graphrag_index_python.py'.",
                    "",
                )

            # Initialize custom models
            logger.info("Initializing custom models for GraphRAG search...")
            from src.graphrag_models import (
                AnthropicChatModel,
                HFEmbeddingModel,
                register_models,
            )

            register_models()

            self.chat_model = AnthropicChatModel(model=self.chat_model_name)
            self.embedding_model = HFEmbeddingModel(model_name=self.embedding_model_name)

            logger.info(f"  ✓ Chat model: {self.chat_model_name}")
            logger.info(f"  ✓ Embedding model: {self.embedding_model_name}")

            # Load parquet files
            logger.info(f"Loading GraphRAG index from {self.output_dir}")

            self.entities = pd.read_parquet(
                self.output_dir / "create_final_entities.parquet"
            )
            self.relationships = pd.read_parquet(
                self.output_dir / "create_final_relationships.parquet"
            )
            self.communities = pd.read_parquet(
                self.output_dir / "create_final_communities.parquet"
            )
            self.community_reports = pd.read_parquet(
                self.output_dir / "create_final_community_reports.parquet"
            )
            self.text_units = pd.read_parquet(
                self.output_dir / "create_final_text_units.parquet"
            )

            logger.info(
                f"Loaded {len(self.entities)} entities, "
                f"{len(self.relationships)} relationships, "
                f"{len(self.communities)} communities, "
                f"{len(self.text_units)} text units"
            )

            # Build lookup indices
            self._build_lookup_indices()

            self.is_indexed = True
            logger.info("GraphRAG index loaded successfully")

        except RetrievalError:
            raise
        except Exception as e:
            raise RetrievalError(
                "graphrag", f"Failed to load GraphRAG index: {str(e)}", ""
            )

    def _build_lookup_indices(self) -> None:
        """Build lookup indices for fast entity retrieval."""
        if self.entities is None:
            return

        self._entity_name_to_id = {}
        self._entity_id_to_data = {}

        for _, row in self.entities.iterrows():
            entity_id = row.get("id", "")
            entity_name = row.get("name", "").lower()

            if entity_id and entity_name:
                self._entity_name_to_id[entity_name] = entity_id
                self._entity_id_to_data[entity_id] = row.to_dict()

        logger.debug(f"Built lookup indices for {len(self._entity_id_to_data)} entities")

    def retrieve(self, query_text: str, top_k: int = 5) -> RetrievalResult:
        """
        Retrieve relevant passages using GraphRAG Python API.

        Args:
            query_text: Query string
            top_k: Number of results to return (overrides config if provided)

        Returns:
            RetrievalResult with ranked passages

        Raises:
            IndexNotBuiltError: If index not loaded
            RetrievalError: If retrieval fails
        """
        if not self.is_indexed:
            raise IndexNotBuiltError("graphrag")

        start_time = time.time()

        try:
            # Use config top_k if not overridden
            k = top_k if top_k != 5 else self.top_k

            # Perform GraphRAG search based on method
            if self.method == "local":
                retrieved_passages = self._local_search(query_text, k)
                strategy = "graphrag_local"
            else:  # global
                retrieved_passages = self._global_search(query_text, k)
                strategy = "graphrag_global"

            retrieval_time = time.time() - start_time

            return RetrievalResult(
                query_id="",  # Will be set by caller
                retrieved_passages=retrieved_passages,
                strategy=strategy,
                retrieval_time_seconds=retrieval_time,
                metadata={
                    "method": self.method,
                    "community_level": (
                        self.community_level if self.method == "global" else None
                    ),
                    "top_k": k,
                    "response_type": self.response_type,
                    "chat_model": self.chat_model_name,
                    "embedding_model": self.embedding_model_name,
                },
            )

        except Exception as e:
            raise RetrievalError(
                f"graphrag_{self.method}",
                f"GraphRAG {self.method} search failed: {str(e)}",
                query_text,
            )

    def _local_search(self, query_text: str, top_k: int) -> List[RetrievedPassage]:
        """
        Perform local search (entity-focused retrieval).

        Local search:
        1. Extract entities mentioned in the query
        2. Find related entities and their neighborhoods
        3. Retrieve relevant text units
        4. Generate a synthesized response

        Args:
            query_text: Query string
            top_k: Number of results

        Returns:
            List of RetrievedPassage objects
        """
        logger.info(f"Performing GraphRAG local search: {query_text[:50]}...")

        if (
            self.entities is None
            or self.text_units is None
            or self.chat_model is None
            or self.embedding_model is None
        ):
            logger.error("GraphRAG index not properly loaded")
            return []

        try:
            # Step 1: Find relevant entities using embedding similarity
            relevant_entities = self._find_relevant_entities(query_text, top_k * 2)

            # Step 2: Get text units for relevant entities
            relevant_text_units = self._get_entity_text_units(relevant_entities)

            # Step 3: Score and rank text units
            passages = self._create_passages_from_text_units(
                relevant_text_units, query_text, "local", top_k
            )

            logger.info(f"Local search returned {len(passages)} passages")
            return passages

        except Exception as e:
            logger.error(f"Local search failed: {e}")
            return []

    def _global_search(self, query_text: str, top_k: int) -> List[RetrievedPassage]:
        """
        Perform global search (community-level retrieval).

        Global search:
        1. Find relevant communities based on query
        2. Use community summaries to answer high-level questions
        3. Aggregate information from community reports

        Args:
            query_text: Query string
            top_k: Number of results

        Returns:
            List of RetrievedPassage objects
        """
        logger.info(f"Performing GraphRAG global search: {query_text[:50]}...")

        if self.community_reports is None or self.chat_model is None:
            logger.error("GraphRAG index not properly loaded")
            return []

        try:
            # Step 1: Find relevant community reports
            relevant_reports = self._find_relevant_community_reports(
                query_text, top_k * 2
            )

            # Step 2: Create passages from community reports
            passages = self._create_passages_from_reports(
                relevant_reports, query_text, top_k
            )

            logger.info(f"Global search returned {len(passages)} passages")
            return passages

        except Exception as e:
            logger.error(f"Global search failed: {e}")
            return []

    def _find_relevant_entities(
        self, query_text: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Find entities relevant to the query using embedding similarity."""
        if self.entities is None or self.embedding_model is None:
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.embed([query_text])[0]

        # Score entities by embedding similarity
        scored_entities = []

        for _, row in self.entities.iterrows():
            entity_embedding = row.get("description_embedding")

            if entity_embedding is not None and isinstance(entity_embedding, list):
                # Calculate cosine similarity
                score = self._cosine_similarity(query_embedding, entity_embedding)
                scored_entities.append(
                    {
                        "id": row.get("id"),
                        "name": row.get("name"),
                        "type": row.get("type"),
                        "description": row.get("description"),
                        "text_unit_ids": row.get("text_unit_ids", []),
                        "score": score,
                    }
                )
            else:
                # Fallback: simple text matching
                name = str(row.get("name", "")).lower()
                desc = str(row.get("description", "")).lower()
                query_lower = query_text.lower()

                if name in query_lower or any(
                    word in desc for word in query_lower.split()
                ):
                    scored_entities.append(
                        {
                            "id": row.get("id"),
                            "name": row.get("name"),
                            "type": row.get("type"),
                            "description": row.get("description"),
                            "text_unit_ids": row.get("text_unit_ids", []),
                            "score": 0.5,
                        }
                    )

        # Sort by score and return top results
        scored_entities.sort(key=lambda x: x["score"], reverse=True)
        return scored_entities[:limit]

    def _get_entity_text_units(
        self, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get text units associated with entities."""
        if self.text_units is None:
            return []

        # Collect all text unit IDs from entities
        text_unit_ids = set()
        for entity in entities:
            tu_ids = entity.get("text_unit_ids", [])
            if isinstance(tu_ids, list):
                text_unit_ids.update(tu_ids)

        # Fetch text units
        text_units = []
        for _, row in self.text_units.iterrows():
            tu_id = row.get("id")
            if tu_id in text_unit_ids:
                text_units.append(
                    {
                        "id": tu_id,
                        "text": row.get("text", ""),
                        "n_tokens": row.get("n_tokens", 0),
                    }
                )

        return text_units

    def _find_relevant_community_reports(
        self, query_text: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Find community reports relevant to the query."""
        if self.community_reports is None or self.embedding_model is None:
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.embed([query_text])[0]

        # Score reports by similarity
        scored_reports = []

        for _, row in self.community_reports.iterrows():
            summary = row.get("summary", "")
            title = row.get("title", "")

            # Generate embedding for summary
            if summary:
                report_embedding = self.embedding_model.embed([summary])[0]
                score = self._cosine_similarity(query_embedding, report_embedding)
            else:
                # Fallback text matching
                query_lower = query_text.lower()
                if any(word in title.lower() for word in query_lower.split()):
                    score = 0.3
                else:
                    score = 0.1

            scored_reports.append(
                {
                    "id": row.get("id"),
                    "community_id": row.get("community_id"),
                    "title": title,
                    "summary": summary,
                    "full_content": row.get("full_content", summary),
                    "rank": row.get("rank", 1.0),
                    "level": row.get("level", 0),
                    "score": score,
                }
            )

        # Sort by score and return top results
        scored_reports.sort(key=lambda x: x["score"], reverse=True)
        return scored_reports[:limit]

    def _create_passages_from_text_units(
        self,
        text_units: List[Dict[str, Any]],
        query_text: str,
        method: str,
        top_k: int,
    ) -> List[RetrievedPassage]:
        """Create RetrievedPassage objects from text units."""
        passages = []

        for idx, tu in enumerate(text_units[:top_k]):
            text = tu.get("text", "")
            if not text:
                continue

            passage_id = f"graphrag_{method}_{tu.get('id', idx)}"

            evidence_passage = EvidencePassage(
                id=passage_id,
                text=text,
                document_id=f"graphrag_{method}",
                metadata={
                    "source": f"graphrag_{method}_search",
                    "query": query_text,
                    "method": method,
                    "text_unit_id": tu.get("id"),
                },
            )

            retrieved_passage = RetrievedPassage(
                passage=evidence_passage,
                score=1.0 / (idx + 1),  # Descending score by rank
                rank=idx + 1,
                is_relevant=None,
            )
            passages.append(retrieved_passage)

        return passages

    def _create_passages_from_reports(
        self,
        reports: List[Dict[str, Any]],
        query_text: str,
        top_k: int,
    ) -> List[RetrievedPassage]:
        """Create RetrievedPassage objects from community reports."""
        passages = []

        for idx, report in enumerate(reports[:top_k]):
            # Use full content if available, otherwise summary
            text = report.get("full_content") or report.get("summary", "")
            if not text:
                continue

            passage_id = f"graphrag_global_{report.get('id', idx)}"

            evidence_passage = EvidencePassage(
                id=passage_id,
                text=text,
                document_id="graphrag_global",
                metadata={
                    "source": "graphrag_global_search",
                    "query": query_text,
                    "method": "global",
                    "community_id": report.get("community_id"),
                    "title": report.get("title"),
                    "level": report.get("level"),
                },
            )

            # Use report's score if available
            score = report.get("score", 1.0 / (idx + 1))

            retrieved_passage = RetrievedPassage(
                passage=evidence_passage,
                score=float(score),
                rank=idx + 1,
                is_relevant=None,
            )
            passages.append(retrieved_passage)

        return passages

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np

        v1 = np.array(vec1)
        v2 = np.array(vec2)

        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))
