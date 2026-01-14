"""
Dense embedding-based retrieval using sentence transformers and ChromaDB.

This module implements dense retrieval with semantic search capabilities.

NOTE: This retrieval strategy is NOT actively tested in the current workflow.
The codebase focuses on BM25 retrieval for baseline experiments.
If you need dense retrieval, you may need to update dependencies and test thoroughly.
"""

from typing import List, Optional
import time
import numpy as np
from src.retrieval.base import RetrieverBase, IndexNotBuiltError, RetrievalError
from src.data_handler.models import EvidencePassage, RetrievalResult, RetrievedPassage


class DenseRetriever(RetrieverBase):
    """Dense embedding-based retrieval using sentence transformers."""

    def __init__(self, config: dict):
        """
        Initialize dense retriever.

        Args:
            config: Configuration dict with:
                - embedding_model: Model name (default: "all-MiniLM-L6-v2")
                - collection_name: ChromaDB collection name
                - similarity_metric: "cosine" or "dot_product"
        """
        super().__init__(config)
        self.embedding_model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
        self.collection_name = config.get("collection_name", "finder_corpus")
        self.similarity_metric = config.get("similarity_metric", "cosine")

        try:
            from sentence_transformers import SentenceTransformer
            import chromadb
            from chromadb.config import Settings

            self.embedding_model = SentenceTransformer(self.embedding_model_name)

            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path="./data/embeddings",
                settings=Settings(anonymized_telemetry=False),
            )

            self.collection = None
            self.passages = []

        except ImportError as e:
            raise ImportError(
                f"Required packages not installed: {e}. "
                "Install with: pip install sentence-transformers chromadb"
            )

    def index_corpus(self, passages: List[EvidencePassage]) -> None:
        """Build dense embedding index in ChromaDB."""
        try:
            self.passages = passages

            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.similarity_metric},
            )

            # Check if already indexed
            if self.collection.count() == len(passages):
                print(f"✓ Dense index already exists with {len(passages)} passages")
                self.is_indexed = True
                return

            print(f"Building dense index for {len(passages)} passages...")

            # Generate embeddings in batches
            batch_size = 32
            passage_texts = [p.text for p in passages]
            passage_ids = [p.id for p in passages]
            passage_metadata = [
                {
                    "document_id": p.document_id,
                    "source_type": p.metadata.source_type,
                }
                for p in passages
            ]

            for i in range(0, len(passages), batch_size):
                batch_texts = passage_texts[i : i + batch_size]
                batch_ids = passage_ids[i : i + batch_size]
                batch_metadata = passage_metadata[i : i + batch_size]

                # Generate embeddings
                embeddings = self.embedding_model.encode(
                    batch_texts,
                    show_progress_bar=False,
                ).tolist()

                # Add to ChromaDB
                self.collection.add(
                    documents=batch_texts,
                    embeddings=embeddings,
                    ids=batch_ids,
                    metadatas=batch_metadata,
                )

                if (i + batch_size) % 100 == 0:
                    print(
                        f"  Indexed {min(i + batch_size, len(passages))}/{len(passages)} passages"
                    )

            self.is_indexed = True
            print(f"✓ Dense index built with {len(passages)} passages")

        except Exception as e:
            raise RetrievalError("dense", f"Failed to build index: {e}", "")

    def retrieve(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """Retrieve top-K passages using dense semantic search."""
        if not self.is_indexed or self.collection is None:
            raise IndexNotBuiltError("dense")

        start_time = time.time()

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(
                query_text,
                show_progress_bar=False,
            ).tolist()

            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
            )

            # Parse results
            retrieved_passages = []
            if results["ids"] and results["ids"][0]:
                for rank, (passage_id, distance) in enumerate(
                    zip(results["ids"][0], results["distances"][0]), start=1
                ):
                    # Find passage by ID
                    passage = next(
                        (p for p in self.passages if p.id == passage_id), None
                    )

                    if passage is None:
                        continue

                    # Convert distance to similarity score (ChromaDB returns distances)
                    score = (
                        1.0 - distance
                        if self.similarity_metric == "cosine"
                        else distance
                    )

                    retrieved_passage = RetrievedPassage(
                        passage=passage,
                        score=float(score),
                        rank=rank,
                        is_relevant=None,
                    )
                    retrieved_passages.append(retrieved_passage)

            retrieval_time = time.time() - start_time

            return RetrievalResult(
                query_id=query_text,
                retrieved_passages=retrieved_passages,
                strategy="dense",
                retrieval_time_seconds=retrieval_time,
                metadata={
                    "embedding_model": self.embedding_model_name,
                    "similarity_metric": self.similarity_metric,
                },
            )

        except Exception as e:
            raise RetrievalError("dense", f"Retrieval failed: {e}", query_text)
