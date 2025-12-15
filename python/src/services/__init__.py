"""Service layer for oceanic-project-management.

Provides business logic services for projects, tasks, and work orders.
Services coordinate between repositories and external integrations.
"""

from .embedding_service import (
    EmbeddingService,
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    VoyageEmbeddingProvider,
    CohereEmbeddingProvider,
    EchoModelServerProvider,
    create_embedding_service,
)

__all__ = [
    "EmbeddingService",
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "VoyageEmbeddingProvider",
    "CohereEmbeddingProvider",
    "EchoModelServerProvider",
    "create_embedding_service",
]
