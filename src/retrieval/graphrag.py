"""
GraphRAG knowledge graph-based retrieval using microsoft/graphrag SDK.

This module implements GraphRAGRetriever which uses Microsoft GraphRAG's
built-in LocalSearch and GlobalSearch engines instead of custom implementations.
Uses adapter classes to make Anthropic and HuggingFace models compatible with
GraphRAG's ChatModel and EmbeddingModel protocols.
"""

import os
import time
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer

# GraphRAG SDK imports
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.config.models.vector_store_schema_config import VectorStoreSchemaConfig
from graphrag.query import factory as search_factory
from graphrag.vector_stores.lancedb import LanceDBVectorStore

from src.retrieval.base import RetrieverBase, RetrievalError, IndexNotBuiltError
from src.data_handler.models import EvidencePassage, RetrievalResult, RetrievedPassage

logger = logging.getLogger(__name__)


class GraphRAGRetriever(RetrieverBase):
    """
    GraphRAG-based retrieval using microsoft/graphrag SDK.

    This retriever loads a pre-built GraphRAG index and uses the SDK's
    LocalSearch and GlobalSearch engines for retrieval.

    Two query modes:
    - local: Entity-focused retrieval (specific entities and neighborhoods)
    - global: Community-level retrieval (high-level summaries)

    Uses adapter classes to make Anthropic and HuggingFace models compatible
    with GraphRAG's ChatModel and EmbeddingModel protocols.

    Note: Requires pre-built GraphRAG index in the specified output directory.
    Build index with: python scripts/graphrag_index.py
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GraphRAG retriever.

        Args:
            config: Configuration dict with keys:
                - graphrag_root: Path to GraphRAG directory (default: "./ragtest")
                - method: Query method "local" or "global" (default: "local")
                - top_k: Number of results to return (default: 5)
                - community_level: Community level for global search (default: 2)
                - response_type: Response type (default: "Multiple Paragraphs")
                - chat_model: Anthropic model name (default: "claude-3-5-haiku-20241022")
                - embedding_model: HuggingFace model name (default: "all-MiniLM-L6-v2")
        """
        super().__init__(config)

        self.graphrag_root = Path(config.get("graphrag_root", "./ragtest"))
        self.method = config.get("method", "local").lower()
        self.top_k = config.get("top_k", 5)
        self.community_level = config.get("community_level", 2)
        self.response_type = config.get("response_type", "Multiple Paragraphs")

        # Model configuration
        self.chat_model_name = config.get("chat_model", "claude-3-5-haiku-20241022")
        self.embedding_model_name = config.get(
            "embedding_model", "text-embedding-3-small"
        )

        # Validate method
        if self.method not in ["local", "global"]:
            raise ValueError(
                f"Invalid method: {self.method}. Must be 'local' or 'global'"
            )

        # Paths
        self.output_dir = self.graphrag_root / "output"
        self.lancedb_path = self.output_dir / "lancedb"

        # Loaded data (initialized in index_corpus)
        self.entities = None
        self.relationships = None
        self.community_reports = None
        self.text_units = None
        self.communities = None

        # GraphRAG config (initialized in index_corpus)
        self.graphrag_config: Optional[GraphRagConfig] = None
        self.vector_store: Optional[LanceDBVectorStore] = None

        # Search engine (initialized in index_corpus)
        self.search_engine = None

        logger.info(
            f"Initialized GraphRAGRetriever: method={self.method}, "
            f"chat_model={self.chat_model_name}, embedding_model={self.embedding_model_name}"
        )

    def index_corpus(self, passages: List[EvidencePassage]) -> None:
        """
        Load pre-built GraphRAG index from parquet files.

        This method loads an existing index built by the GraphRAG CLI.
        It does NOT build a new index from passages.

        Args:
            passages: Evidence passages (not used - index already built)

        Raises:
            RetrievalError: If GraphRAG index not found or loading fails
        """
        try:
            logger.info("===  Starting index_corpus ===")
            # Check if output directory exists
            logger.info(f"Checking output directory: {self.output_dir}")
            if not self.output_dir.exists():
                raise RetrievalError(
                    "graphrag",
                    f"GraphRAG output directory not found at {self.output_dir}. "
                    "Run 'make graphrag-index' or 'python scripts/graphrag_index.py' first.",
                    "",
                )

            # Check for required parquet files (CORRECT NAMES)
            required_files = [
                "entities.parquet",
                "relationships.parquet",
                "communities.parquet",
                "community_reports.parquet",
                "text_units.parquet",
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
                    "Index may be incomplete. Re-run 'make graphrag-index'.",
                    "",
                )

            # Load parquet files
            logger.info(f"Loading GraphRAG index from {self.output_dir}")

            logger.info("  Loading entities.parquet...")
            entities_df = pd.read_parquet(self.output_dir / "entities.parquet")
            logger.info("  Loading relationships.parquet...")
            relationships_df = pd.read_parquet(
                self.output_dir / "relationships.parquet"
            )
            logger.info("  Loading communities.parquet...")
            communities_df = pd.read_parquet(self.output_dir / "communities.parquet")
            logger.info("  Loading community_reports.parquet...")
            reports_df = pd.read_parquet(self.output_dir / "community_reports.parquet")
            logger.info("  Loading text_units.parquet...")
            text_units_df = pd.read_parquet(self.output_dir / "text_units.parquet")

            logger.info(
                f"Loaded {len(entities_df)} entities, "
                f"{len(relationships_df)} relationships, "
                f"{len(communities_df)} communities, "
                f"{len(reports_df)} community reports, "
                f"{len(text_units_df)} text units"
            )

            # Convert DataFrames to GraphRAG models using SDK adapters
            logger.info("Converting DataFrames to GraphRAG models...")

            self.entities = read_indexer_entities(
                entities_df, communities_df, self.community_level
            )
            self.relationships = read_indexer_relationships(relationships_df)
            self.community_reports = read_indexer_reports(
                reports_df, communities_df, self.community_level
            )
            self.text_units = read_indexer_text_units(text_units_df)
            self.communities = communities_df  # Keep DataFrame for factory functions

            logger.info("  ✓ Converted to GraphRAG entity/relationship models")

            # Initialize vector store (LanceDB)
            if self.lancedb_path.exists():
                logger.info(
                    f"Initializing LanceDB vector store from {self.lancedb_path}"
                )

                # Create VectorStoreSchemaConfig for the new API
                # Note: index_name must match actual LanceDB table name
                schema_config = VectorStoreSchemaConfig(
                    index_name="default-entity-description",
                    id_field="id",
                    text_field="text",
                    vector_field="vector",
                    attributes_field="attributes",
                )

                # Initialize and connect to LanceDB
                self.vector_store = LanceDBVectorStore(
                    vector_store_schema_config=schema_config
                )
                self.vector_store.connect(db_uri=str(self.lancedb_path))
                logger.info("  ✓ LanceDB vector store initialized")
            else:
                logger.warning(
                    f"LanceDB directory not found at {self.lancedb_path}. "
                    "Vector similarity search may not work optimally."
                )
                self.vector_store = None

            # Create minimal GraphRAG config for model initialization
            logger.info("Creating GraphRAG config for models...")
            self._create_graphrag_config()

            # Build search engine based on method
            if self.method == "local":
                self._build_local_search_engine()
            else:
                self._build_global_search_engine()

            self.is_indexed = True
            logger.info("GraphRAG index loaded successfully")

        except RetrievalError:
            raise
        except Exception as e:
            logger.error(f"Failed to load GraphRAG index: {e}", exc_info=True)
            raise RetrievalError(
                "graphrag", f"Failed to load GraphRAG index: {str(e)}", ""
            )

    def _create_graphrag_config(self) -> None:
        """
        Create a minimal GraphRAG config for model initialization.

        Auto-detects provider based on model name (OpenAI vs Anthropic).
        """
        # Detect provider based on model name
        if self.chat_model_name.startswith("gpt-"):
            provider = "openai"
            api_key = os.getenv("OPENAI_API_KEY")
        elif self.chat_model_name.startswith("claude-"):
            provider = "anthropic"
            api_key = os.getenv("ANTHROPIC_API_KEY")
        else:
            # Default to openai for other models
            provider = "openai"
            api_key = os.getenv("OPENAI_API_KEY")
            logger.warning(
                f"Unknown model prefix for {self.chat_model_name}, defaulting to OpenAI provider"
            )

        # Create chat model config
        chat_model_config = LanguageModelConfig(
            model_provider=provider,
            model=self.chat_model_name,
            deployment_name=self.chat_model_name,
            api_key=api_key,
            type="openai_chat",
            max_tokens=4096,
            temperature=0.0,
        )
        embedding_model_config = LanguageModelConfig(
            model_provider=provider,
            model=self.embedding_model_name,
            deployment_name=self.embedding_model_name,
            api_key=api_key,
            type="openai_embedding",
        )

        # Create minimal GraphRagConfig
        self.graphrag_config = GraphRagConfig(
            root_dir=str(self.graphrag_root),
            models={
                "default_chat_model": chat_model_config,
                "default_embedding_model": embedding_model_config,
            },
        )

        logger.info(
            f"  ✓ Created GraphRAG config with {provider}/{self.chat_model_name}"
        )

    def _build_local_search_engine(self) -> None:
        """
        Build LocalSearch engine using GraphRAG factory function.

        LocalSearch provides entity-focused retrieval with:
        - Entity text embeddings for similarity search
        - Text units associated with entities
        - Relationship context
        - Community context
        """
        logger.info("Building LocalSearch engine...")

        # Use factory function to create search engine
        self.search_engine = search_factory.get_local_search_engine(
            config=self.graphrag_config,
            reports=self.community_reports,
            text_units=self.text_units,
            entities=self.entities,
            relationships=self.relationships,
            covariates={},  # Empty covariates for now
            description_embedding_store=self.vector_store,
            response_type=self.response_type,
        )

        logger.info("  ✓ LocalSearch engine built")

    def _build_global_search_engine(self) -> None:
        """
        Build GlobalSearch engine using GraphRAG factory function.

        GlobalSearch provides community-level retrieval with:
        - Community reports (hierarchical summaries)
        - Map-reduce approach for comprehensive answers
        - Multi-level community context
        """
        logger.info("Building GlobalSearch engine...")

        # Convert communities DataFrame to list of Community objects
        from graphrag.data_model.community import Community

        communities_list = [
            Community(
                id=str(row["id"]),
                level=int(row.get("level", 0)),
                title=str(row.get("title", "")),
            )
            for _, row in self.communities.iterrows()
        ]

        # Use factory function to create search engine
        self.search_engine = search_factory.get_global_search_engine(
            config=self.graphrag_config,
            reports=self.community_reports,
            entities=self.entities,
            communities=communities_list,
            response_type=self.response_type,
        )

        logger.info("  ✓ GlobalSearch engine built")

    def retrieve(self, query_text: str, top_k: int = 5) -> RetrievalResult:
        """
        Retrieve relevant passages using GraphRAG SDK search engines.

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

            # Perform GraphRAG search using SDK (async wrapped in sync)
            logger.info(
                f"Performing GraphRAG {self.method} search: {query_text[:50]}..."
            )

            search_result = self.search_engine.search(query=query_text)

            # Convert GraphRAG SearchResult to List[RetrievedPassage]
            retrieved_passages = self._convert_to_passages(
                search_result=search_result,
                query_text=query_text,
                top_k=k,
            )

            retrieval_time = time.time() - start_time

            logger.info(
                f"GraphRAG {self.method} search returned {len(retrieved_passages)} passages "
                f"in {retrieval_time:.2f}s"
            )

            # Generate query_id from query text hash for uniqueness
            import hashlib

            query_hash = hashlib.md5(query_text.encode()).hexdigest()[:8]
            query_id = f"graphrag_{query_hash}"

            return RetrievalResult(
                query_id=query_id,
                retrieved_passages=retrieved_passages,
                strategy=f"graphrag_{self.method}",
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
                    "has_context_data": hasattr(search_result, "context_data"),
                },
            )

        except Exception as e:
            logger.error(f"GraphRAG {self.method} search failed: {e}", exc_info=True)
            raise RetrievalError(
                f"graphrag_{self.method}",
                f"GraphRAG {self.method} search failed: {str(e)}",
                query_text,
            )

    def _convert_to_passages(
        self,
        search_result: Any,
        query_text: str,
        top_k: int,
    ) -> List[RetrievedPassage]:
        """
        Convert GraphRAG SearchResult to List[RetrievedPassage].

        GraphRAG returns a synthesized response, not individual passages.
        We extract source passages from context_data if available, or use
        the synthesized response as a single passage.

        Args:
            search_result: GraphRAG SearchResult object
            query_text: Original query text
            top_k: Number of passages to return

        Returns:
            List of RetrievedPassage objects
        """
        passages = []

        # Extract response text
        response_text = getattr(search_result, "response", "")
        context_data = getattr(search_result, "context_data", {})

        # Try to extract source passages from context_data
        if context_data and isinstance(context_data, dict):
            # Local search may have 'sources' or 'text_units'
            # Global search may have 'reports' or 'community_reports'

            sources = context_data.get("sources", [])
            reports = context_data.get("reports", [])

            # Handle both list and DataFrame types for sources/reports
            has_sources = sources is not None and (
                (isinstance(sources, list) and len(sources) > 0)
                or (hasattr(sources, "empty") and not sources.empty)
            )
            has_reports = reports is not None and (
                (isinstance(reports, list) and len(reports) > 0)
                or (hasattr(reports, "empty") and not reports.empty)
            )

            if self.method == "local" and has_sources:
                # Extract text units from local search context
                for idx, source in enumerate(sources[:top_k]):
                    if isinstance(source, dict):
                        text = source.get("text", "")
                        source_id = source.get("id", f"local_{idx}")
                    else:
                        # If source is a string, use it directly
                        text = str(source)
                        source_id = f"local_{idx}"

                    if not text.strip():
                        continue

                    evidence_passage = EvidencePassage(
                        id=f"graphrag_local_{source_id}",
                        text=text,
                        document_id="graphrag_local",
                        metadata={
                            "source": "graphrag_local_search",
                            "query": query_text,
                            "method": "local",
                        },
                    )

                    passages.append(
                        RetrievedPassage(
                            passage=evidence_passage,
                            score=1.0 / (idx + 1),  # Descending score by rank
                            rank=idx + 1,
                            is_relevant=None,
                        )
                    )

            elif self.method == "global" and has_reports:
                # Extract community reports from global search context
                for idx, report in enumerate(reports[:top_k]):
                    if isinstance(report, dict):
                        text = report.get("content", "") or report.get("summary", "")
                        report_id = report.get("id", f"global_{idx}")
                    else:
                        text = str(report)
                        report_id = f"global_{idx}"

                    if not text.strip():
                        continue

                    evidence_passage = EvidencePassage(
                        id=f"graphrag_global_{report_id}",
                        text=text,
                        document_id="graphrag_global",
                        metadata={
                            "source": "graphrag_global_search",
                            "query": query_text,
                            "method": "global",
                        },
                    )

                    passages.append(
                        RetrievedPassage(
                            passage=evidence_passage,
                            score=1.0 / (idx + 1),
                            rank=idx + 1,
                            is_relevant=None,
                        )
                    )

        # Fallback: Use synthesized response as single passage
        if not passages and response_text.strip():
            logger.warning(
                "No source passages found in context_data. "
                "Using synthesized response as single passage."
            )

            evidence_passage = EvidencePassage(
                id=f"graphrag_{self.method}_response",
                text=response_text,
                document_id=f"graphrag_{self.method}",
                metadata={
                    "source": f"graphrag_{self.method}_search",
                    "query": query_text,
                    "method": self.method,
                    "note": "Synthesized response from GraphRAG (no individual source passages)",
                },
            )

            passages.append(
                RetrievedPassage(
                    passage=evidence_passage,
                    score=1.0,
                    rank=1,
                    is_relevant=None,
                )
            )

        return passages
