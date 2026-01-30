"""
LLM-based passage reranking using OpenAI structured outputs.

This module provides reranking functionality that uses an LLM to score
retrieved passages for relevance to a query, then filters to top-k.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from src.data_handler.models import RetrievedPassage


class PassageScore(BaseModel):
    """Single passage with relevance score (simple, no reasoning)."""

    index: int = Field(description="Index of the passage (0-based)")
    score: float = Field(
        description="Relevance score 0.0-1.0", ge=0.0, le=1.0
    )


class RerankedPassages(BaseModel):
    """Reranked passages from LLM - simple scores only."""

    scores: List[PassageScore] = Field(
        description="Relevance scores for each passage"
    )


def build_reranking_prompt(query_text: str, passages: List[RetrievedPassage]) -> str:
    """
    Build the prompt for LLM-based reranking.

    Args:
        query_text: The user's query
        passages: List of retrieved passages to rerank

    Returns:
        Formatted prompt string for the reranker LLM
    """
    passage_texts = []
    for i, p in enumerate(passages):
        # Truncate to 500 chars to keep prompt manageable
        text = p.passage.text[:500]
        if len(p.passage.text) > 500:
            text += "..."
        passage_texts.append(f"[{i}] {text}")

    passages_str = "\n\n".join(passage_texts)

    prompt = f"""Rate each passage's relevance to this financial query (0.0-1.0 scale).

Query: {query_text}

Passages:
{passages_str}

Scoring guide:
- 1.0: Directly answers the question
- 0.7-0.9: Highly relevant context
- 0.4-0.6: Somewhat relevant
- 0.1-0.3: Marginally relevant
- 0.0: Irrelevant

Return a score for each passage index."""

    return prompt


def rerank_passages(
    query_text: str,
    passages: List[RetrievedPassage],
    reranker_llm,
    rerank_top_k: Optional[int] = None,
) -> List[RetrievedPassage]:
    """
    Rerank passages using LLM-based scoring and filter to top-k.

    Args:
        query_text: The user's query
        passages: List of retrieved passages to rerank
        reranker_llm: LLM client with generate_structured() method
        rerank_top_k: Number of passages to keep after reranking (None = keep all)

    Returns:
        Reranked and filtered list of passages with updated scores and ranks
    """
    if not passages:
        return passages

    # Build the reranking prompt
    prompt = build_reranking_prompt(query_text, passages)

    # Get structured scores from LLM
    result = reranker_llm.generate_structured(prompt, RerankedPassages)

    # Create a mapping from index to score
    score_map = {ps.index: ps.score for ps in result.scores}

    # Update passage scores based on LLM reranking
    reranked = []
    for i, passage in enumerate(passages):
        if i in score_map:
            # Create a new RetrievedPassage with updated score
            # Note: We preserve original passage data but update the score
            reranked.append(
                RetrievedPassage(
                    passage=passage.passage,
                    score=score_map[i],
                    rank=passage.rank,  # Will be updated after sorting
                    is_relevant=passage.is_relevant,
                )
            )
        else:
            # If LLM didn't return a score for this passage, keep original
            reranked.append(passage)

    # Sort by new scores (descending)
    reranked.sort(key=lambda p: p.score, reverse=True)

    # Filter to top-k if specified
    if rerank_top_k is not None and rerank_top_k > 0:
        reranked = reranked[:rerank_top_k]

    # Update ranks
    for i, passage in enumerate(reranked):
        passage.rank = i + 1

    return reranked
