"""
Index builder utilities for creating and loading retrieval indexes.

This module provides utilities for building and persisting BM25 indexes.
Note: Dense and hybrid retrieval are not actively tested. BM25 is the primary retrieval strategy.
"""

from pathlib import Path
from typing import List
import pickle
from src.data_handler.models import EvidencePassage
from src.retrieval.bm25 import BM25Retriever


class IndexBuilder:
    """Builder for creating and persisting retrieval indexes."""

    def __init__(self, index_dir: Path = Path("./data/embeddings")):
        """
        Initialize index builder.

        Args:
            index_dir: Directory to store indexes
        """
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def build_bm25_index(
        self,
        passages: List[EvidencePassage],
        output_path: Path,
    ) -> None:
        """
        Build and save BM25 index.

        Args:
            passages: List of evidence passages to index
            output_path: Path to save serialized index
        """
        print(f"Building BM25 index for {len(passages)} passages...")

        config = {"k1": 1.5, "b": 0.75}
        retriever = BM25Retriever(config)
        retriever.index_corpus(passages)

        # Save index and passages
        index_data = {
            "bm25": retriever.bm25,
            "passages": passages,
            "tokenized_corpus": retriever.tokenized_corpus,
            "config": config,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(index_data, f)

        print(f"✓ BM25 index saved to {output_path}")


    def load_bm25_index(self, index_path: Path) -> BM25Retriever:
        """
        Load pre-built BM25 index.

        Args:
            index_path: Path to serialized index

        Returns:
            Initialized BM25Retriever with loaded index

        Raises:
            FileNotFoundError: If index file doesn't exist
        """
        if not index_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {index_path}")

        with open(index_path, "rb") as f:
            index_data = pickle.load(f)

        config = index_data["config"]
        retriever = BM25Retriever(config)
        retriever.bm25 = index_data["bm25"]
        retriever.passages = index_data["passages"]
        retriever.tokenized_corpus = index_data["tokenized_corpus"]
        retriever.is_indexed = True

        print(f"✓ BM25 index loaded from {index_path}")
        return retriever
