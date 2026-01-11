"""
Retry logic with exponential backoff and jitter for API calls.

This module provides decorators and utilities for retrying failed operations
with intelligent backoff strategies to handle transient failures and rate limits.
"""

import time
import random
import logging
from typing import Callable, Any, Optional, Type, Tuple
from functools import wraps

from .exceptions import RateLimitError, ConnectionError as LangfuseConnectionError
from .config import get_retry_settings

logger = logging.getLogger(__name__)


def calculate_backoff_delay(
    attempt: int, base_delay: float = 1.0, max_delay: float = 60.0, jitter: bool = True
) -> float:
    """
    Calculate exponential backoff delay with optional jitter.

    Args:
        attempt: Current retry attempt number (1-indexed)
        base_delay: Base delay in seconds for first retry
        max_delay: Maximum delay in seconds
        jitter: Whether to add random jitter (recommended to prevent thundering herd)

    Returns:
        Delay in seconds before next retry

    Example:
        >>> calculate_backoff_delay(1)  # ~1 second
        1.234
        >>> calculate_backoff_delay(3)  # ~4 seconds with jitter
        4.567
        >>> calculate_backoff_delay(10)  # capped at max_delay
        60.0
    """
    # Exponential backoff: base_delay * 2^(attempt-1)
    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)

    # Add jitter: random value between 0 and delay
    if jitter:
        delay = delay * (0.5 + random.random() * 0.5)  # 50%-100% of calculated delay

    return delay


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (RateLimitError, LangfuseConnectionError),
    jitter: bool = True,
    log_errors: bool = True,
) -> Callable:
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for first retry
        max_delay: Maximum delay in seconds
        exceptions: Tuple of exception types to retry on
        jitter: Whether to add random jitter to delays
        log_errors: Whether to log retry attempts

    Returns:
        Decorated function with retry logic

    Example:
        >>> @retry_with_backoff(max_retries=3)
        ... def upload_item(item):
        ...     client.create_dataset_item(item)
        >>>
        >>> upload_item(my_item)  # Will retry up to 3 times on rate limit/connection errors
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            attempt = 0
            last_exception = None

            while attempt <= max_retries:
                try:
                    result = func(*args, **kwargs)

                    # Log successful retry
                    if attempt > 0 and log_errors:
                        logger.info(
                            f"{func.__name__} succeeded after {attempt} retry(ies)"
                        )

                    return result

                except exceptions as e:
                    last_exception = e
                    attempt += 1

                    if attempt > max_retries:
                        # Max retries exceeded
                        if log_errors:
                            logger.error(
                                f"{func.__name__} failed after {max_retries} retries: {e}"
                            )
                        raise

                    # Calculate delay and wait
                    delay = calculate_backoff_delay(
                        attempt=attempt,
                        base_delay=base_delay,
                        max_delay=max_delay,
                        jitter=jitter,
                    )

                    if log_errors:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt}/{max_retries}): {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )

                    time.sleep(delay)

                except Exception as e:
                    # Non-retryable exception, fail immediately
                    if log_errors:
                        logger.error(
                            f"{func.__name__} failed with non-retryable error: {e}"
                        )
                    raise

            # Should not reach here, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


def retry_on_rate_limit(func: Callable) -> Callable:
    """
    Decorator specifically for retrying on Langfuse rate limits.

    Uses retry settings from config (langfuse.yaml) with exponential backoff
    and jitter optimized for API rate limits.

    Args:
        func: Function to decorate

    Returns:
        Decorated function with rate limit retry logic

    Example:
        >>> @retry_on_rate_limit
        ... def create_dataset(client, name):
        ...     return client.create_dataset(name)
    """
    try:
        settings = get_retry_settings()
    except Exception:
        # Fallback to defaults if config loading fails
        settings = {"max_retries": 3, "retry_base_delay": 1.0, "retry_max_delay": 60.0}

    return retry_with_backoff(
        max_retries=settings["max_retries"],
        base_delay=settings["retry_base_delay"],
        max_delay=settings["retry_max_delay"],
        exceptions=(RateLimitError,),
        jitter=True,
        log_errors=True,
    )(func)


def retry_on_connection_error(func: Callable) -> Callable:
    """
    Decorator specifically for retrying on connection errors.

    Uses shorter delays than rate limit retries since connection errors
    may resolve more quickly.

    Args:
        func: Function to decorate

    Returns:
        Decorated function with connection error retry logic

    Example:
        >>> @retry_on_connection_error
        ... def fetch_dataset(client, name):
        ...     return client.get_dataset(name)
    """
    return retry_with_backoff(
        max_retries=3,
        base_delay=0.5,
        max_delay=10.0,
        exceptions=(LangfuseConnectionError,),
        jitter=True,
        log_errors=True,
    )(func)


class RetryContext:
    """
    Context manager for retrying a block of code with exponential backoff.

    Useful for retry logic in imperative code that doesn't fit the decorator pattern.

    Example:
        >>> retry_ctx = RetryContext(max_retries=3)
        >>> for attempt in retry_ctx:
        ...     try:
        ...         result = some_api_call()
        ...         break  # Success, exit retry loop
        ...     except RateLimitError as e:
        ...         if not retry_ctx.should_retry(e):
        ...             raise
        ...         # Will automatically sleep before next attempt
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
    ):
        """
        Initialize RetryContext.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            jitter: Whether to add random jitter
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.attempt = 0
        self.last_exception = None

    def __iter__(self):
        """Iterate through retry attempts."""
        self.attempt = 0
        return self

    def __next__(self):
        """Get next retry attempt."""
        if self.attempt > self.max_retries:
            raise StopIteration

        attempt_num = self.attempt
        self.attempt += 1

        # Sleep before retry (except first attempt)
        if attempt_num > 0:
            delay = calculate_backoff_delay(
                attempt=attempt_num,
                base_delay=self.base_delay,
                max_delay=self.max_delay,
                jitter=self.jitter,
            )
            logger.debug(f"Retry attempt {attempt_num}, sleeping {delay:.2f}s")
            time.sleep(delay)

        return attempt_num

    def should_retry(self, exception: Exception) -> bool:
        """
        Check if should retry given an exception.

        Args:
            exception: Exception that was raised

        Returns:
            True if should retry, False if max retries exceeded
        """
        self.last_exception = exception

        if self.attempt > self.max_retries:
            logger.error(f"Max retries ({self.max_retries}) exceeded: {exception}")
            return False

        return True
