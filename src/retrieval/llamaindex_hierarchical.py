"""
Hierarchical Chunking with Auto-Merging Retriever using LlamaIndex.

This module wraps LlamaIndex's HierarchicalNodeParser and AutoMergingRetriever
to provide hierarchical chunking within our RetrieverBase interface.
"""

import time
from typing import List, Optional

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.retrieval.base import RetrieverBase, IndexNotBuiltError, RetrievalError
from src.data_handler.models import (
    EvidencePassage,
    RetrievalResult,
    RetrievedPassage,
    EvidenceMetadata,
)


class LlamaIndexHierarchicalRetriever(RetrieverBase):
    """Hierarchical chunking with auto-merging using LlamaIndex."""

    def __init__(self, config: dict):
        """
        Initialize hierarchical retriever.

        Args:
            config: Configuration dict with hierarchical parameters
                - hierarchical_chunk_sizes: List[int] (default [2048, 512, 128])
                - hierarchical_chunk_overlap: int (default 20)
                - hierarchical_merge_threshold: float (default 0.5)
                - embedding_model: str (default "all-MiniLM-L6-v2")
                - top_k: int (default 5)
        """
        super().__init__(config)

        # Extract configuration
        self.chunk_sizes = config.get("hierarchical_chunk_sizes", [2048, 512, 128])
        self.chunk_overlap = config.get("hierarchical_chunk_overlap", 20)
        self.merge_threshold = config.get("hierarchical_merge_threshold", 0.5)
        self.embedding_model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
        self.top_k_default = config.get("top_k", 5)

        # LlamaIndex components (initialized during indexing)
        self.parser: Optional[HierarchicalNodeParser] = None
        self.index: Optional[VectorStoreIndex] = None
        self.retriever: Optional[AutoMergingRetriever] = None
        self.storage_context: Optional[StorageContext] = None
        self.embed_model: Optional[HuggingFaceEmbedding] = None

    def index_corpus(self, passages: List[EvidencePassage]) -> None:
        """
        Build hierarchical index using LlamaIndex.

        Steps:
        1. Convert EvidencePassage → LlamaIndex Document
        2. Parse documents with HierarchicalNodeParser
        3. Build VectorStoreIndex from nodes
        4. Create AutoMergingRetriever

        Args:
            passages: Evidence passages to index

        Raises:
            RetrievalError: If indexing fails
        """
        try:
            start_time = time.time()

            # Step 1: Initialize embedding model
            self.embed_model = HuggingFaceEmbedding(model_name=self.embedding_model_name)

            # Step 2: Convert passages to LlamaIndex Documents
            documents = [
                Document(
                    text=passage.text,
                    doc_id=passage.id,
                    metadata={
                        "passage_id": passage.id,
                        "document_id": passage.document_id,
                        "source_type": passage.metadata.source_type,
                    },
                )
                for passage in passages
            ]

            # Step 3: Create hierarchical node parser
            self.parser = HierarchicalNodeParser.from_defaults(
                chunk_sizes=self.chunk_sizes, chunk_overlap=self.chunk_overlap
            )

            # Step 4: Parse documents into hierarchical nodes
            nodes = self.parser.get_nodes_from_documents(documents)

            # Step 5: Create storage context
            self.storage_context = StorageContext.from_defaults()
            self.storage_context.docstore.add_documents(nodes)

            # Step 6: Build vector index (only indexes leaf nodes automatically)
            self.index = VectorStoreIndex(
                nodes=nodes,
                storage_context=self.storage_context,
                embed_model=self.embed_model,
                show_progress=True,
            )

            # Step 7: Create auto-merging retriever
            base_retriever = self.index.as_retriever(
                similarity_top_k=self.top_k_default * 3  # Retrieve more for merging
            )

            self.retriever = AutoMergingRetriever(
                base_retriever,
                self.storage_context,
                verbose=True,
                simple_ratio_thresh=self.merge_threshold,
            )

            self.is_indexed = True

            elapsed = time.time() - start_time
            print(
                f"Indexed {len(passages)} passages with hierarchical chunking in {elapsed:.2f}s"
            )
            print(f"  Chunk sizes: {self.chunk_sizes}")
            print(f"  Total nodes: {len(nodes)}")
            print(f"  Merge threshold: {self.merge_threshold}")

        except Exception as e:
            raise RetrievalError("hierarchical", f"Indexing failed: {str(e)}", "")

    def retrieve(self, query_text: str, top_k: int = 5) -> RetrievalResult:
        """
        Retrieve with auto-merging.

        Steps:
        1. Call LlamaIndex AutoMergingRetriever
        2. Convert NodeWithScore → RetrievedPassage
        3. Return RetrievalResult

        Args:
            query_text: Query string
            top_k: Number of passages to retrieve

        Returns:
            RetrievalResult with merged passages

        Raises:
            IndexNotBuiltError: If index not built
            RetrievalError: If retrieval fails
        """
        if not self.is_indexed:
            raise IndexNotBuiltError("hierarchical")

        try:
            start_time = time.time()

            # Step 1: Retrieve with auto-merging
            nodes_with_scores = self.retriever.retrieve(query_text)

            # Step 2: Convert to RetrievedPassage
            retrieved_passages = []
            for rank, node_with_score in enumerate(nodes_with_scores[:top_k]):
                # Extract node and score
                node = node_with_score.node
                score = node_with_score.score

                # Create EvidencePassage
                evidence_passage = EvidencePassage(
                    id=node.node_id,
                    text=node.get_content(),
                    document_id=node.metadata.get("document_id", "unknown"),
                    metadata=EvidenceMetadata(
                        source_type=node.metadata.get(
                            "source_type", "hierarchical_chunk"
                        ),
                        publication_date=None,
                        page_number=None,
                        section=None,
                    ),
                )

                # Create RetrievedPassage
                retrieved_passage = RetrievedPassage(
                    passage=evidence_passage,
                    score=float(score) if score is not None else 0.0,
                    rank=rank + 1,
                    is_relevant=None,  # Set by evaluation
                )

                retrieved_passages.append(retrieved_passage)

            elapsed = time.time() - start_time

            # Step 3: Return result
            return RetrievalResult(
                query_id="",
                retrieved_passages=retrieved_passages,
                strategy="hierarchical",
                retrieval_time_seconds=elapsed,
                metadata={
                    "chunk_sizes": self.chunk_sizes,
                    "merge_threshold": self.merge_threshold,
                    "embedding_model": self.embedding_model_name,
                    "top_k": top_k,
                    "nodes_retrieved": len(nodes_with_scores),
                },
            )

        except Exception as e:
            raise RetrievalError(
                "hierarchical", f"Retrieval failed: {str(e)}", query_text
            )
