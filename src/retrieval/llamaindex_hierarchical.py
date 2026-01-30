"""
Hierarchical Chunking with Auto-Merging Retriever using LlamaIndex.

This module wraps LlamaIndex's HierarchicalNodeParser and AutoMergingRetriever
to provide hierarchical chunking within our RetrieverBase interface.
"""

import hashlib
import time
from pathlib import Path
from typing import List, Optional

from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
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
                - persist_dir: str (default "./data/hierarchical_index")
        """
        super().__init__(config)

        # Extract configuration
        self.chunk_sizes = config.get("hierarchical_chunk_sizes", [2048, 512, 128])
        self.chunk_overlap = config.get("hierarchical_chunk_overlap", 20)
        self.merge_threshold = config.get("hierarchical_merge_threshold", 0.5)
        self.embedding_model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
        self.top_k_default = config.get("top_k", 5)
        self.persist_dir = config.get("persist_dir", "./data/hierarchical_index")

        # Reranking configuration
        self.enable_reranking = config.get("enable_reranking", False)
        self.rerank_top_k = config.get("rerank_top_k", None)
        self.reranker_llm = config.get("reranker_llm", None)

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
            print(f"[1/7] Initializing embedding model: {self.embedding_model_name}...")
            self.embed_model = HuggingFaceEmbedding(
                model_name=self.embedding_model_name
            )

            # Step 2: Convert passages to LlamaIndex Documents
            print(
                f"[2/7] Converting {len(passages)} passages to LlamaIndex Documents..."
            )
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
            print(
                f"[3/7] Creating hierarchical node parser (chunk sizes: {self.chunk_sizes})..."
            )
            self.parser = HierarchicalNodeParser.from_defaults(
                chunk_sizes=self.chunk_sizes, chunk_overlap=self.chunk_overlap
            )

            # Step 4: Parse documents into hierarchical nodes
            print("[4/7] Parsing documents into hierarchical nodes...")
            nodes = self.parser.get_nodes_from_documents(documents)

            # Calculate node statistics by level
            leaf_nodes = [n for n in nodes if not n.child_nodes]
            parent_nodes = [n for n in nodes if n.child_nodes]
            root_nodes = [n for n in nodes if not n.parent_node]

            print(f"       Total nodes: {len(nodes)}")
            print(f"       Root nodes (level 0): {len(root_nodes)}")
            print(f"       Parent nodes (intermediate): {len(parent_nodes)}")
            print(f"       Leaf nodes (indexed): {len(leaf_nodes)}")

            # Step 5: Create storage context
            print("[5/7] Creating storage context and adding nodes to docstore...")
            self.storage_context = StorageContext.from_defaults()
            self.storage_context.docstore.add_documents(nodes)

            # Step 6: Build vector index (only indexes leaf nodes automatically)
            print("[6/7] Building vector index (embedding leaf nodes)...")
            self.index = VectorStoreIndex(
                nodes=nodes,
                storage_context=self.storage_context,
                embed_model=self.embed_model,
                show_progress=True,
            )

            # Step 7: Create auto-merging retriever
            print("[7/7] Creating auto-merging retriever...")
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

            # Calculate average text lengths
            avg_leaf_len = (
                sum(len(n.get_content()) for n in leaf_nodes) / len(leaf_nodes)
                if leaf_nodes
                else 0
            )
            avg_parent_len = (
                sum(len(n.get_content()) for n in parent_nodes) / len(parent_nodes)
                if parent_nodes
                else 0
            )

            print("\n" + "=" * 60)
            print("HIERARCHICAL INDEX COMPLETE")
            print("=" * 60)
            print(f"  Time elapsed: {elapsed:.2f}s")
            print(f"  Input passages: {len(passages)}")
            print(f"  Chunk sizes: {self.chunk_sizes}")
            print(f"  Chunk overlap: {self.chunk_overlap}")
            print(f"  Merge threshold: {self.merge_threshold}")
            print("-" * 60)
            print("  Node Statistics:")
            print(f"    Total nodes: {len(nodes)}")
            print(f"    Root nodes: {len(root_nodes)}")
            print(f"    Parent nodes: {len(parent_nodes)}")
            print(f"    Leaf nodes (indexed): {len(leaf_nodes)}")
            print(f"    Expansion ratio: {len(nodes) / len(passages):.2f}x")
            print("-" * 60)
            print("  Average Content Length:")
            print(f"    Leaf nodes: {avg_leaf_len:.0f} chars")
            print(f"    Parent nodes: {avg_parent_len:.0f} chars")
            print("=" * 60 + "\n")

            # Auto-persist after indexing
            self.save_index()

        except Exception as e:
            raise RetrievalError("hierarchical", f"Indexing failed: {str(e)}", "")

    def _get_persist_path(self) -> Path:
        """
        Generate persist path based on config hash.

        The hash includes chunk sizes, overlap, and embedding model to ensure
        different configurations use different persist directories.

        Returns:
            Path to the persist directory for this configuration
        """
        config_str = (
            f"{self.chunk_sizes}_{self.chunk_overlap}_{self.embedding_model_name}"
        )
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        return Path(self.persist_dir) / config_hash

    def load_index(self) -> bool:
        """
        Load persisted index if it exists.

        Returns:
            True if index was loaded successfully, False otherwise
        """
        persist_path = self._get_persist_path()
        if not persist_path.exists():
            return False

        try:
            start_time = time.time()

            # Initialize embedding model
            self.embed_model = HuggingFaceEmbedding(
                model_name=self.embedding_model_name
            )

            # Load storage context from disk
            self.storage_context = StorageContext.from_defaults(
                persist_dir=str(persist_path)
            )

            # Rebuild VectorStoreIndex from storage context
            self.index = load_index_from_storage(
                self.storage_context, embed_model=self.embed_model
            )

            # Create auto-merging retriever
            base_retriever = self.index.as_retriever(
                similarity_top_k=self.top_k_default * 3
            )
            self.retriever = AutoMergingRetriever(
                base_retriever,
                self.storage_context,
                verbose=True,
                simple_ratio_thresh=self.merge_threshold,
            )

            self.is_indexed = True

            elapsed = time.time() - start_time
            print(f"Loaded hierarchical index from {persist_path} in {elapsed:.2f}s")
            return True

        except Exception as e:
            print(f"Failed to load index from {persist_path}: {e}")
            return False

    def save_index(self) -> None:
        """
        Persist the index to disk.

        Raises:
            IndexNotBuiltError: If index hasn't been built yet
        """
        if not self.is_indexed:
            raise IndexNotBuiltError("hierarchical")

        persist_path = self._get_persist_path()
        persist_path.mkdir(parents=True, exist_ok=True)

        # Persist storage context (includes docstore and vector store)
        self.storage_context.persist(persist_dir=str(persist_path))
        print(f"Index persisted to {persist_path}")

    def index_corpus_or_load(self, passages: List[EvidencePassage]) -> None:
        """
        Load existing index or build and persist new one.

        This is the recommended method to use for experiments as it avoids
        rebuilding the index if a compatible one already exists.

        Args:
            passages: Evidence passages to index (used only if building new index)
        """
        if self.load_index():
            print(f"Using existing hierarchical index from {self._get_persist_path()}")
            return

        print("No existing index found, building new index...")
        self.index_corpus(passages)

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

            retrieval_elapsed = time.time() - start_time

            # Step 3: Optional reranking
            reranking_metadata = {}
            if self.enable_reranking and self.reranker_llm is not None:
                from src.retrieval.reranker import rerank_passages

                rerank_start = time.time()
                retrieved_passages = rerank_passages(
                    query_text=query_text,
                    passages=retrieved_passages,
                    reranker_llm=self.reranker_llm,
                    rerank_top_k=self.rerank_top_k,
                )
                reranking_elapsed = time.time() - rerank_start

                reranking_metadata = {
                    "reranking_enabled": True,
                    "reranking_latency_seconds": reranking_elapsed,
                    "passages_before_rerank": top_k,
                    "passages_after_rerank": len(retrieved_passages),
                }

            total_elapsed = time.time() - start_time

            # Step 4: Return result
            metadata = {
                "chunk_sizes": self.chunk_sizes,
                "merge_threshold": self.merge_threshold,
                "embedding_model": self.embedding_model_name,
                "top_k": top_k,
                "nodes_retrieved": len(nodes_with_scores),
                "retrieval_latency_seconds": retrieval_elapsed,
            }
            metadata.update(reranking_metadata)

            return RetrievalResult(
                query_id="a",
                retrieved_passages=retrieved_passages,
                strategy="hierarchical",
                retrieval_time_seconds=total_elapsed,
                metadata=metadata,
            )

        except Exception as e:
            raise RetrievalError(
                "hierarchical", f"Retrieval failed: {str(e)}", query_text
            )
