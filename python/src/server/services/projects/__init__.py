"""
Projects Services Package

This package contains all services related to project management,
including project CRUD operations, task management, document management,
versioning, progress tracking, source linking, and AI-assisted project creation.

Service Factory:
    Use get_project_service() and get_task_service() to get the appropriate
    service implementation based on STORAGE_BACKEND environment variable.

    STORAGE_BACKEND=supabase (default): Use Supabase-backed services
    STORAGE_BACKEND=vespa: Use Vespa-backed services with hybrid search
"""

from .document_service import DocumentService
from .project_creation_service import ProjectCreationService
from .project_service import ProjectService
from .source_linking_service import SourceLinkingService
from .task_service import TaskService
from .versioning_service import VersioningService

# Service factory functions - use these for backend-agnostic code
from .service_factory import (
    get_project_service,
    get_task_service,
    get_storage_backend,
    is_vespa_enabled,
    get_backend_info,
)

# Vespa implementations (can be imported directly if needed)
from .vespa_project_service import VespaProjectService
from .vespa_task_service import VespaTaskService

__all__ = [
    # Original Supabase services
    "ProjectService",
    "TaskService",
    "DocumentService",
    "VersioningService",
    "ProjectCreationService",
    "SourceLinkingService",
    # Service factory
    "get_project_service",
    "get_task_service",
    "get_storage_backend",
    "is_vespa_enabled",
    "get_backend_info",
    # Vespa services
    "VespaProjectService",
    "VespaTaskService",
]
