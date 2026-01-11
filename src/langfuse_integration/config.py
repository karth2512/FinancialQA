"""
Configuration and credential management for Langfuse integration.

This module handles loading configuration from files and environment variables,
with fail-fast validation to prevent silent failures.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class CredentialError(Exception):
    """Raised when Langfuse credentials are missing or invalid."""

    pass


class ConfigurationError(Exception):
    """Raised when configuration is invalid or incomplete."""

    pass


def validate_langfuse_credentials(
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    host: Optional[str] = None,
    fail_fast: bool = True,
) -> Dict[str, str]:
    """
    Validate Langfuse credentials are available and properly formatted.

    Args:
        public_key: Langfuse public key (if not provided, reads from env)
        secret_key: Langfuse secret key (if not provided, reads from env)
        host: Langfuse host URL (if not provided, reads from env)
        fail_fast: Whether to raise error immediately on missing credentials

    Returns:
        Dict with validated credentials (public_key, secret_key, host)

    Raises:
        CredentialError: If credentials are missing or invalid and fail_fast=True

    Example:
        >>> credentials = validate_langfuse_credentials()
        >>> client = Langfuse(**credentials)
    """
    # Load environment variables from .env file if present
    load_dotenv()

    # Retrieve credentials from environment if not provided
    if public_key is None:
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")

    if secret_key is None:
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if host is None:
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    # Validate credentials are present
    errors = []

    if not public_key:
        errors.append(
            "LANGFUSE_PUBLIC_KEY is missing. "
            "Set it in your .env file or environment variables."
        )

    if not secret_key:
        errors.append(
            "LANGFUSE_SECRET_KEY is missing. "
            "Set it in your .env file or environment variables."
        )

    # Validate public key format
    if public_key and not public_key.startswith("pk-lf-"):
        errors.append(
            f"LANGFUSE_PUBLIC_KEY has invalid format. "
            f"Expected to start with 'pk-lf-', got: {public_key[:10]}..."
        )

    # Validate secret key format
    if secret_key and not secret_key.startswith("sk-lf-"):
        errors.append(
            f"LANGFUSE_SECRET_KEY has invalid format. "
            f"Expected to start with 'sk-lf-', got: {secret_key[:10]}..."
        )

    # Validate host URL format
    if host and not host.startswith(("http://", "https://")):
        errors.append(
            f"LANGFUSE_HOST has invalid format. " f"Expected HTTP(S) URL, got: {host}"
        )

    if errors and fail_fast:
        error_message = "\n".join(
            [
                "Langfuse credential validation failed:",
                "",
                *[f"  - {error}" for error in errors],
                "",
                "Get your credentials from: https://cloud.langfuse.com → Settings → API Keys",
                "Then add them to your .env file or environment variables.",
            ]
        )
        raise CredentialError(error_message)

    return {
        "public_key": public_key or "",
        "secret_key": secret_key or "",
        "host": host or "https://cloud.langfuse.com",
    }


def is_tracing_enabled() -> bool:
    """
    Check if Langfuse tracing is enabled via environment variable.

    Returns:
        True if tracing is enabled (default), False otherwise

    Example:
        >>> if is_tracing_enabled():
        ...     client = Langfuse()
        ... else:
        ...     print("Tracing disabled, skipping Langfuse initialization")
    """
    load_dotenv()

    enabled = os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower()

    return enabled in {"true", "1", "yes", "on"}


def load_langfuse_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load Langfuse configuration from YAML file.

    Args:
        config_path: Path to langfuse.yaml configuration file.
                     If None, uses default: src/config/langfuse.yaml

    Returns:
        Dict with configuration settings

    Raises:
        ConfigurationError: If config file is missing or invalid

    Example:
        >>> config = load_langfuse_config()
        >>> batch_size = config.get("upload_batch_size", 50)
    """
    if config_path is None:
        # Default path relative to project root
        config_path = Path(__file__).parent.parent / "config" / "langfuse.yaml"

    if not config_path.exists():
        raise ConfigurationError(
            f"Langfuse configuration file not found: {config_path}\n"
            f"Expected location: src/config/langfuse.yaml"
        )

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ConfigurationError(f"Config file is empty: {config_path}")

        return config

    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"Failed to parse Langfuse config file: {config_path}\n" f"Error: {e}"
        )


def get_flush_settings(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get flush settings for Langfuse client from config.

    Args:
        config: Configuration dict (if None, loads from default path)

    Returns:
        Dict with flush_interval and flush_batch_size

    Example:
        >>> flush_settings = get_flush_settings()
        >>> client = Langfuse(**credentials, **flush_settings)
    """
    if config is None:
        config = load_langfuse_config()

    return {
        "flush_interval": config.get("flush_interval", 5.0),
        "flush_at": config.get("flush_batch_size", 100),
    }


def get_retry_settings(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get retry configuration for API calls.

    Args:
        config: Configuration dict (if None, loads from default path)

    Returns:
        Dict with max_retries, retry_base_delay, retry_max_delay

    Example:
        >>> retry_settings = get_retry_settings()
        >>> max_retries = retry_settings["max_retries"]
    """
    if config is None:
        config = load_langfuse_config()

    return {
        "max_retries": config.get("max_retries", 3),
        "retry_base_delay": config.get("retry_base_delay", 1.0),
        "retry_max_delay": config.get("retry_max_delay", 60.0),
    }


def get_upload_settings(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get dataset upload configuration.

    Args:
        config: Configuration dict (if None, loads from default path)

    Returns:
        Dict with upload_batch_size and upload_rate_limit_delay

    Example:
        >>> upload_settings = get_upload_settings()
        >>> batch_size = upload_settings["upload_batch_size"]
    """
    if config is None:
        config = load_langfuse_config()

    return {
        "upload_batch_size": config.get("upload_batch_size", 50),
        "upload_rate_limit_delay": config.get("upload_rate_limit_delay", 0.5),
    }
