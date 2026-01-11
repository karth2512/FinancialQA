"""
FinDER dataset loader using HuggingFace datasets library.

This module handles downloading, caching, and loading the FinDER dataset
with version tracking for reproducibility.
"""

import sys
from datasets import load_dataset
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime
from src.data_handler.models import (
    Query,
    QueryMetadata,
    EvidencePassage,
    EvidenceMetadata,
)

# Configure stdout to use UTF-8 encoding on Windows
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


class FinDERLoader:
    """Loader for the FinDER financial QA dataset."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        revision: str = "main",
    ):
        """
        Initialize FinDER dataset loader.

        Args:
            cache_dir: Directory to cache downloaded dataset (default: ./data/finder)
            revision: Git revision/commit to pin for reproducibility
        """
        self.cache_dir = cache_dir or Path("./data/finder")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.revision = revision
        self.dataset = None
        self.metadata_path = self.cache_dir / "metadata.json"

    def download(self) -> None:
        """
        Download FinDER dataset from HuggingFace Hub.

        Saves metadata including download timestamp and dataset version.

        Raises:
            DatasetLoadError: If download fails
        """
        try:
            print(f"Downloading FinDER dataset (revision: {self.revision})...")
            self.dataset = load_dataset(
                "Linq-AI-Research/FinDER",
                revision=self.revision,
                cache_dir=str(self.cache_dir),
            )

            # Save metadata for provenance tracking
            metadata = {
                "dataset_name": "Linq-AI-Research/FinDER",
                "revision": self.revision,
                "download_timestamp": datetime.utcnow().isoformat(),
                "dataset_version": self.revision,
                "num_queries": len(self.dataset["train"]) if self.dataset else 0,
            }

            with open(self.metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"✓ Downloaded {metadata['num_queries']} queries")
            print(f"✓ Metadata saved to {self.metadata_path}")

        except Exception as e:
            raise DatasetLoadError(f"Failed to download FinDER dataset: {e}")

    def load(self) -> List[Query]:
        """
        Load FinDER dataset and convert to Query objects.

        Returns:
            List of Query objects with metadata

        Raises:
            DatasetLoadError: If dataset not downloaded or load fails
        """
        if self.dataset is None:
            try:
                self.dataset = load_dataset(
                    "Linq-AI-Research/FinDER",
                    revision=self.revision,
                    cache_dir=str(self.cache_dir),
                )
            except Exception as e:
                raise DatasetLoadError(
                    f"Dataset not found. Run download() first. Error: {e}"
                )

        queries = []
        dataset_split = self.dataset["train"]  # FinDER uses train split

        for idx, example in enumerate(dataset_split):
            try:
                query = self._parse_example(idx, example)
                queries.append(query)
            except Exception as e:
                print(f"Warning: Failed to parse query {idx}: {e}")
                continue

        return queries

    def _parse_example(self, idx: int, example: Dict[str, Any]) -> Query:
        """
        Parse a single dataset example into a Query object.

        Args:
            idx: Example index
            example: Raw example dict from dataset

        Returns:
            Parsed Query object
        """
        # FinDER dataset structure
        # Fields: _id, text (query), reasoning, category, references (evidence), answer, type
        query_text = example.get("text", "")
        expected_answer = example.get("answer", "")
        expected_evidence = example.get("references", [])

        if isinstance(expected_evidence, str):
            expected_evidence = [expected_evidence]

        # Extract metadata from dataset fields
        reasoning_required = example.get("reasoning", False)
        category = example.get("category", "unknown")
        query_type = example.get("type", "unknown")

        metadata = QueryMetadata(
            domain_term_count=self._count_domain_terms(query_text),
            has_ambiguity=self._detect_ambiguity(query_text),
            query_type=query_type,
            required_evidence_count=len(expected_evidence) if expected_evidence else 1,
            financial_subdomain=category,
            reasoning_required=reasoning_required,
        )

        # Use dataset's _id if available, otherwise use index
        query_id = example.get("_id", f"finder_{idx}")

        return Query(
            id=query_id,
            text=query_text,
            expected_answer=expected_answer,
            expected_evidence=expected_evidence,
            metadata=metadata,
        )

    def load_corpus(self) -> List[EvidencePassage]:
        """
        Load evidence corpus for retrieval indexing.

        Returns:
            List of EvidencePassage objects

        Raises:
            DatasetLoadError: If corpus not available
        """
        if self.dataset is None:
            self.load()

        passages = []
        dataset_split = self.dataset["train"]

        # Extract unique passages from references (evidence)
        seen_passages = set()
        passage_id = 0

        for example in dataset_split:
            # FinDER uses "references" field for evidence passages
            evidence_list = example.get("references", [])
            if isinstance(evidence_list, str):
                evidence_list = [evidence_list]

            for evidence_text in evidence_list:
                if evidence_text and evidence_text not in seen_passages:
                    seen_passages.add(evidence_text)

                    metadata = EvidenceMetadata(
                        source_type="unknown",  # Would need to parse from dataset
                        publication_date=None,
                        page_number=None,
                        section=None,
                    )

                    passage = EvidencePassage(
                        id=f"passage_{passage_id}",
                        text=evidence_text,
                        document_id=f"doc_{passage_id}",
                        metadata=metadata,
                    )

                    passages.append(passage)
                    passage_id += 1

        return passages

    def get_dataset_version(self) -> str:
        """
        Get dataset version from metadata.

        Returns:
            Dataset version string

        Raises:
            DatasetLoadError: If metadata not found
        """
        if not self.metadata_path.exists():
            raise DatasetLoadError("Dataset metadata not found. Run download() first.")

        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)

        return metadata.get("dataset_version", "unknown")

    def _count_domain_terms(self, text: str) -> int:
        """Count financial domain-specific terms in text."""
        # Simple heuristic: count uppercase abbreviations and financial keywords
        financial_keywords = {
            "CAGR",
            "P/E",
            "EBITDA",
            "ROE",
            "ROI",
            "EPS",
            "revenue",
            "earnings",
            "profit",
            "margin",
            "equity",
            "debt",
            "dividend",
            "MS",
            "GS",
            "JPM",
            "trading",
            "investment",
            "portfolio",
        }

        words = text.split()
        count = sum(1 for word in words if word in financial_keywords)
        return count

    def _detect_ambiguity(self, text: str) -> bool:
        """Detect if query contains ambiguous abbreviations."""
        ambiguous_terms = {"MS", "GS", "JPM", "DB", "CS", "recent", "current"}
        words = text.split()
        return any(word in ambiguous_terms for word in words)

    def _classify_query_type(self, text: str) -> str:
        """Classify query type based on keywords."""
        text_lower = text.lower()

        if any(word in text_lower for word in ["what is", "define", "meaning"]):
            return "definition"
        elif any(word in text_lower for word in ["compare", "versus", "vs", "than"]):
            return "temporal_comparison"
        elif any(
            word in text_lower for word in ["trend", "growth", "increase", "decrease"]
        ):
            return "trend_analysis"
        else:
            return "factual"


class DatasetLoadError(Exception):
    """Raised when dataset loading fails."""

    pass


# CLI interface for downloading dataset
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download FinDER dataset")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the dataset",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./data/finder"),
        help="Cache directory for dataset",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Git revision to download",
    )

    args = parser.parse_args()

    loader = FinDERLoader(cache_dir=args.cache_dir, revision=args.revision)

    if args.download:
        loader.download()
        print("\n✓ Dataset download complete!")
        print(f"  Version: {loader.get_dataset_version()}")
        print(f"  Location: {loader.cache_dir}")
    else:
        parser.print_help()
