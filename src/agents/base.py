"""
Base agent interface and data structures for multi-agent RAG system.

This module defines the abstract Agent base class and related data structures
that all concrete agents must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List
from datetime import datetime


@dataclass
class AgentConfig:
    """Configuration for an agent instance."""

    agent_type: str  # "query_understanding", "context_resolution", etc.
    llm_config: Dict[str, Any]  # LLM provider and model configuration
    parameters: Dict[str, Any] = field(
        default_factory=dict
    )  # Agent-specific parameters

    def __post_init__(self):
        """Validate agent configuration."""
        valid_types = {
            "query_understanding",
            "context_resolution",
            "retrieval_strategy",
            "evidence_fusion",
            "answer_synthesis",
        }
        if self.agent_type not in valid_types:
            raise ValueError(f"agent_type must be one of {valid_types}")
        if not self.llm_config:
            raise ValueError("llm_config must be non-empty")


@dataclass
class AgentExecution:
    """Record of a single agent execution."""

    timestamp: str  # ISO 8601
    input_context: Dict[str, Any]  # What the agent received
    output_context: Dict[str, Any]  # What the agent produced
    decisions: List[str]  # Human-readable decision log
    latency_seconds: float  # Execution time
    cost_estimate: float  # Estimated cost in USD

    def __post_init__(self):
        """Validate agent execution record."""
        if self.latency_seconds < 0:
            raise ValueError("latency_seconds must be non-negative")
        if self.cost_estimate < 0:
            raise ValueError("cost_estimate must be non-negative")


class Agent(ABC):
    """Base interface for all agents in the multi-agent system."""

    def __init__(self, config: AgentConfig):
        """
        Initialize agent with configuration.

        Args:
            config: Agent configuration including LLM settings and parameters
        """
        self.config = config
        self.execution_history: List[AgentExecution] = []

    @abstractmethod
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input context and return updated context.

        This is the core method that each agent must implement. The context
        dictionary flows through the agent pipeline, with each agent reading
        inputs and adding outputs.

        Args:
            context: Context dictionary containing:
                - query: Query object (always present)
                - retrieved_passages: List[RetrievedPassage] (if retrieval done)
                - disambiguation: Dict[str, str] (if context resolution done)
                - fused_evidence: List[EvidencePassage] (if fusion done)
                - agent_decisions: List[str] (cumulative decision log)

        Returns:
            Updated context with agent's output added

        Raises:
            AgentExecutionError: If agent processing fails
        """
        pass

    def _record_execution(
        self,
        input_context: Dict[str, Any],
        output_context: Dict[str, Any],
        decisions: List[str],
        latency_seconds: float,
        cost_estimate: float,
    ) -> None:
        """
        Record an agent execution for tracking and debugging.

        Args:
            input_context: What the agent received
            output_context: What the agent produced
            decisions: Human-readable decision log
            latency_seconds: Execution time
            cost_estimate: Estimated cost in USD
        """
        execution = AgentExecution(
            timestamp=datetime.utcnow().isoformat(),
            input_context=input_context.copy(),
            output_context=output_context.copy(),
            decisions=decisions.copy(),
            latency_seconds=latency_seconds,
            cost_estimate=cost_estimate,
        )
        self.execution_history.append(execution)


class AgentExecutionError(Exception):
    """Raised when agent processing fails."""

    def __init__(
        self,
        agent_type: str,
        error_message: str,
        context_snapshot: Dict[str, Any],
    ):
        """
        Initialize agent execution error.

        Args:
            agent_type: Type of agent that failed
            error_message: Human-readable error description
            context_snapshot: Context at time of failure
        """
        self.agent_type = agent_type
        self.error_message = error_message
        self.context_snapshot = context_snapshot
        super().__init__(
            f"Agent '{agent_type}' execution failed: {error_message}"
        )
