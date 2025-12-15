"""Vespa repository layer for oceanic-project-management.

Provides CRUD operations for projects, tasks, and work orders using Vespa
as the backend document store. Replaces Supabase PostgreSQL storage.

Repository classes:
    - VespaClient: Shared HTTP client with connection pooling
    - VespaProjectRepository: Project CRUD with semantic search
    - VespaTaskRepository: Task CRUD with semantic search
    - VespaWorkOrderRepository: Work order CRUD (no embeddings)

Usage:
    >>> from repositories import VespaProjectRepository
    >>> repo = VespaProjectRepository(embedding_service=generate_embedding)
    >>> project = await repo.create({
    ...     "project_id": "proj_123",
    ...     "org_id": "org_456",
    ...     "name": "Q4 Roadmap"
    ... })

Environment Variables:
    VESPA_HOST: Vespa endpoint URL (default: http://localhost:8081)
    VESPA_TIMEOUT: Request timeout in seconds (default: 30)
"""

from .vespa_client import VespaClient
from .vespa_project_repository import VespaProjectRepository
from .vespa_task_repository import VespaTaskRepository
from .vespa_work_order_repository import VespaWorkOrderRepository

__all__ = [
    "VespaClient",
    "VespaProjectRepository",
    "VespaTaskRepository",
    "VespaWorkOrderRepository",
]
