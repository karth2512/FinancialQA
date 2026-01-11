"""
Custom exceptions for Langfuse integration.

This module defines exception types for handling various error conditions
that can occur during dataset management, experiment execution, and API communication.
"""


class LangfuseIntegrationError(Exception):
    """Base exception for all Langfuse integration errors."""

    pass


class CredentialError(LangfuseIntegrationError):
    """Raised when Langfuse credentials are missing, invalid, or unauthorized."""

    pass


class RateLimitError(LangfuseIntegrationError):
    """
    Raised when Langfuse API rate limit is exceeded.

    This error should trigger retry logic with exponential backoff.
    """

    def __init__(
        self,
        message: str = "Langfuse API rate limit exceeded",
        retry_after: float = None,
    ):
        """
        Initialize RateLimitError.

        Args:
            message: Error message
            retry_after: Seconds to wait before retrying (from API response)
        """
        super().__init__(message)
        self.retry_after = retry_after


class DatasetNotFoundError(LangfuseIntegrationError):
    """Raised when a requested Langfuse dataset does not exist."""

    def __init__(self, dataset_name: str):
        """
        Initialize DatasetNotFoundError.

        Args:
            dataset_name: Name of the dataset that was not found
        """
        super().__init__(
            f"Langfuse dataset '{dataset_name}' not found. "
            f"Create it first using upload_finder_dataset() or check the name."
        )
        self.dataset_name = dataset_name


class DatasetUploadError(LangfuseIntegrationError):
    """Raised when dataset upload fails partially or completely."""

    def __init__(
        self,
        message: str,
        total_items: int = 0,
        failed_items: int = 0,
        errors: list = None,
    ):
        """
        Initialize DatasetUploadError.

        Args:
            message: Error message
            total_items: Total number of items attempted
            failed_items: Number of items that failed to upload
            errors: List of error messages from failed uploads
        """
        super().__init__(message)
        self.total_items = total_items
        self.failed_items = failed_items
        self.errors = errors or []


class ExperimentExecutionError(LangfuseIntegrationError):
    """Raised when experiment execution fails critically."""

    def __init__(
        self,
        message: str,
        experiment_name: str = None,
        run_id: str = None,
        partial_results: dict = None,
    ):
        """
        Initialize ExperimentExecutionError.

        Args:
            message: Error message
            experiment_name: Name of the experiment that failed
            run_id: Run ID if available
            partial_results: Partial results if experiment failed mid-execution
        """
        super().__init__(message)
        self.experiment_name = experiment_name
        self.run_id = run_id
        self.partial_results = partial_results or {}


class EvaluatorError(LangfuseIntegrationError):
    """Raised when an evaluator fails to compute scores."""

    def __init__(self, evaluator_name: str, message: str):
        """
        Initialize EvaluatorError.

        Args:
            evaluator_name: Name of the evaluator that failed
            message: Error message explaining the failure
        """
        super().__init__(f"Evaluator '{evaluator_name}' failed: {message}")
        self.evaluator_name = evaluator_name


class ValidationError(LangfuseIntegrationError):
    """Raised when configuration or data validation fails."""

    pass


class ConnectionError(LangfuseIntegrationError):
    """Raised when Langfuse API is unreachable or network errors occur."""

    def __init__(self, message: str = "Failed to connect to Langfuse API"):
        """
        Initialize ConnectionError.

        Args:
            message: Error message
        """
        super().__init__(
            f"{message}\n" f"Check your network connectivity and Langfuse host URL."
        )


class TimeoutError(LangfuseIntegrationError):
    """Raised when an operation times out."""

    def __init__(self, operation: str, timeout_seconds: float):
        """
        Initialize TimeoutError.

        Args:
            operation: Description of the operation that timed out
            timeout_seconds: Timeout duration in seconds
        """
        super().__init__(
            f"Operation '{operation}' timed out after {timeout_seconds} seconds"
        )
        self.operation = operation
        self.timeout_seconds = timeout_seconds
