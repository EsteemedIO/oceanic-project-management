"""
Service Factory Module

Provides factory functions that return the appropriate service implementation
based on the STORAGE_BACKEND environment variable.

Usage:
    from .service_factory import get_project_service, get_task_service

    # Get service based on STORAGE_BACKEND env var
    project_service = get_project_service()

Environment Variables:
    STORAGE_BACKEND: "supabase" (default) or "vespa"
    EMBEDDING_PROVIDER: "openai" (default), "voyage", "cohere", or "echo"
"""

import os
from typing import TYPE_CHECKING

from ...config.logfire_config import get_logger

logger = get_logger(__name__)

# Type hints for IDE support
if TYPE_CHECKING:
    from .project_service import ProjectService
    from .task_service import TaskService
    from .vespa_project_service import VespaProjectService
    from .vespa_task_service import VespaTaskService


def get_storage_backend() -> str:
    """
    Get the configured storage backend.

    Returns:
        "supabase" or "vespa"
    """
    backend = os.getenv("STORAGE_BACKEND", "supabase").lower()
    if backend not in ("supabase", "vespa"):
        logger.warning(f"Invalid STORAGE_BACKEND '{backend}', defaulting to 'supabase'")
        return "supabase"
    return backend


def get_project_service(supabase_client=None):
    """
    Get the project service based on STORAGE_BACKEND configuration.

    Args:
        supabase_client: Optional Supabase client (only used if backend is supabase)

    Returns:
        ProjectService or VespaProjectService instance
    """
    backend = get_storage_backend()

    if backend == "vespa":
        logger.info("Using Vespa backend for ProjectService")
        from .vespa_project_service import VespaProjectService
        return VespaProjectService()
    else:
        logger.debug("Using Supabase backend for ProjectService")
        from .project_service import ProjectService
        return ProjectService(supabase_client)


def get_task_service(supabase_client=None):
    """
    Get the task service based on STORAGE_BACKEND configuration.

    Args:
        supabase_client: Optional Supabase client (only used if backend is supabase)

    Returns:
        TaskService or VespaTaskService instance
    """
    backend = get_storage_backend()

    if backend == "vespa":
        logger.info("Using Vespa backend for TaskService")
        from .vespa_task_service import VespaTaskService
        return VespaTaskService()
    else:
        logger.debug("Using Supabase backend for TaskService")
        from .task_service import TaskService
        return TaskService(supabase_client)


def is_vespa_enabled() -> bool:
    """
    Check if Vespa backend is enabled.

    Returns:
        True if STORAGE_BACKEND is set to "vespa"
    """
    return get_storage_backend() == "vespa"


def get_backend_info() -> dict:
    """
    Get information about the current storage backend configuration.

    Returns:
        Dict with backend info for health checks and debugging
    """
    backend = get_storage_backend()
    info = {
        "storage_backend": backend,
        "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "openai"),
    }

    if backend == "vespa":
        info["vespa_host"] = os.getenv("VESPA_HOST", "localhost")
        info["vespa_port"] = os.getenv("VESPA_PORT", "8081")
    else:
        info["supabase_url"] = os.getenv("SUPABASE_URL", "not_configured")

    return info
