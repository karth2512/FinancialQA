"""
Baseline single-agent RAG pipeline.

This module implements a simple RAG approach: query → retrieve → generate.
"""

import time
from typing import List, Dict, Any
from src.data_handler.models import Query, RetrievalResult
from src.retrieval.base import RetrieverBase
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.utils.llm_client import LLMClient, create_llm_client


class BaselineRAG:
    """Baseline single-agent RAG pipeline."""

    def __init__(
        self,
        retriever: RetrieverBase,
        llm_client: LLMClient,
    ):
        """
        Initialize baseline RAG pipeline.

        Args:
            retriever: Retrieval strategy (BM25, Dense, or Hybrid)
            llm_client: LLM client for answer generation
        """
        self.retriever = retriever
        self.llm_client = llm_client

    def process_query(
        self,
        query: Query,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Process a single query through the RAG pipeline.

        Args:
            query: Query object to process
            top_k: Number of passages to retrieve

        Returns:
            Dict with generated_answer, retrieval_result, latency, cost
        """
        start_time = time.time()

        # Step 1: Retrieve relevant passages
        retrieval_result = self.retriever.retrieve(
            query_text=query.text,
            top_k=top_k,
        )
        retrieval_result.query_id = query.id

        # Step 2: Construct prompt with retrieved evidence
        prompt = self._construct_prompt(query, retrieval_result)

        # Step 3: Generate answer using LLM
        prompt_tokens = self.llm_client.count_tokens(prompt)

        try:
            generated_answer = self.llm_client.generate(prompt)
            completion_tokens = self.llm_client.count_tokens(generated_answer)
        except Exception as e:
            generated_answer = f"Error generating answer: {e}"
            completion_tokens = 0

        # Step 4: Estimate cost
        cost_estimate = self.llm_client.estimate_cost(prompt_tokens, completion_tokens)

        total_latency = time.time() - start_time

        return {
            "generated_answer": generated_answer,
            "retrieval_result": retrieval_result,
            "latency_seconds": total_latency,
            "cost_estimate": cost_estimate,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    def _construct_prompt(
        self,
        query: Query,
        retrieval_result: RetrievalResult,
    ) -> str:
        """
        Construct prompt with query and retrieved evidence.

        Args:
            query: Query object
            retrieval_result: Retrieved passages

        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            "You are a financial question-answering assistant. Answer the following question based on the provided evidence passages.",
            "",
            "Question: " + query.text,
            "",
            "Evidence Passages:",
        ]

        # Add retrieved passages
        for i, retrieved_passage in enumerate(
            retrieval_result.retrieved_passages, start=1
        ):
            prompt_parts.append(
                f"\n[Passage {i}] (Score: {retrieved_passage.score:.3f})"
            )
            prompt_parts.append(retrieved_passage.passage.text)

        prompt_parts.extend(
            [
                "",
                "Instructions:",
                "- Answer the question based ONLY on the evidence passages provided above",
                "- Be concise and factual",
                "- If the evidence is insufficient to answer the question, say 'Insufficient evidence'",
                "- Do not make up information not present in the evidence",
                "",
                "Answer:",
            ]
        )

        return "\n".join(prompt_parts)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaselineRAG":
        """
        Create baseline RAG pipeline from configuration.

        Args:
            config: Configuration dict with:
                - retrieval_config: Retrieval configuration
                - llm_config: LLM configuration

        Returns:
            Initialized BaselineRAG instance
        """
        # Create retriever based on strategy
        retrieval_config = config["retrieval_config"]
        strategy = retrieval_config.get("strategy", "bm25")

        if strategy == "bm25":
            retriever = BM25Retriever(retrieval_config)
        elif strategy == "dense":
            retriever = DenseRetriever(retrieval_config)
        elif strategy == "hybrid":
            retriever = HybridRetriever(retrieval_config)
        else:
            raise ValueError(f"Unsupported retrieval strategy: {strategy}")

        # Create LLM client
        # Get generator config (for baseline, there's only one LLM config)
        llm_configs = config.get("llm_configs", {})
        llm_config = llm_configs.get("generator", llm_configs)

        llm_client = create_llm_client(llm_config)

        return cls(retriever=retriever, llm_client=llm_client)
