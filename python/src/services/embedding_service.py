"""Embedding service for generating vector representations of text.

Supports multiple embedding providers (OpenAI, Cohere, Voyage) with
configurable model selection. Used by Vespa repositories for semantic search.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)

# Default embedding dimensions (matches Vespa schemas)
DEFAULT_EMBEDDING_DIM = 1536


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embedding vector for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (list of floats)
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension for this provider."""
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider using text-embedding-3-small (1536 dimensions)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        timeout: int = 30,
    ) -> None:
        """Initialize OpenAI embedding provider.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model name (default: text-embedding-3-small for 1536d)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required (set OPENAI_API_KEY)")

        self.model = model
        self.timeout = timeout
        self._dimension = 1536  # text-embedding-3-small dimension

        logger.info(
            "openai_embedding_provider_initialized",
            extra={"model": model, "dimension": self._dimension}
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "input": texts,
                    "model": self.model,
                },
            )
            response.raise_for_status()
            data = response.json()

        embeddings = [item["embedding"] for item in data["data"]]

        logger.debug(
            "openai_embeddings_generated",
            extra={"count": len(texts), "model": self.model}
        )

        return embeddings


class VoyageEmbeddingProvider(EmbeddingProvider):
    """Voyage AI embedding provider using voyage-large-2 (1024 dimensions)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "voyage-large-2-instruct",
        timeout: int = 30,
    ) -> None:
        """Initialize Voyage embedding provider.

        Args:
            api_key: Voyage API key (defaults to VOYAGE_API_KEY env var)
            model: Model name (default: voyage-large-2-instruct)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Voyage API key required (set VOYAGE_API_KEY)")

        self.model = model
        self.timeout = timeout
        self._dimension = 1024  # voyage-large-2-instruct dimension

        logger.info(
            "voyage_embedding_provider_initialized",
            extra={"model": model, "dimension": self._dimension}
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                "https://api.voyageai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "input": texts,
                    "model": self.model,
                },
            )
            response.raise_for_status()
            data = response.json()

        embeddings = [item["embedding"] for item in data["data"]]

        logger.debug(
            "voyage_embeddings_generated",
            extra={"count": len(texts), "model": self.model}
        )

        return embeddings


class CohereEmbeddingProvider(EmbeddingProvider):
    """Cohere embedding provider using embed-english-v3.0 (1024 dimensions)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "embed-english-v3.0",
        timeout: int = 30,
    ) -> None:
        """Initialize Cohere embedding provider.

        Args:
            api_key: Cohere API key (defaults to COHERE_API_KEY env var)
            model: Model name (default: embed-english-v3.0)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("Cohere API key required (set COHERE_API_KEY)")

        self.model = model
        self.timeout = timeout
        self._dimension = 1024  # embed-english-v3.0 dimension

        logger.info(
            "cohere_embedding_provider_initialized",
            extra={"model": model, "dimension": self._dimension}
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                "https://api.cohere.ai/v1/embed",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "texts": texts,
                    "model": self.model,
                    "input_type": "search_document",
                },
            )
            response.raise_for_status()
            data = response.json()

        embeddings = data["embeddings"]

        logger.debug(
            "cohere_embeddings_generated",
            extra={"count": len(texts), "model": self.model}
        )

        return embeddings


class EchoModelServerProvider(EmbeddingProvider):
    """Echo model server embedding provider for local/self-hosted models.

    Connects to Echo's model server for embedding generation using
    sentence-transformers models like nomic-embed-text-v1.
    """

    def __init__(
        self,
        model_server_url: Optional[str] = None,
        model_name: str = "nomic-ai/nomic-embed-text-v1",
        timeout: int = 30,
    ) -> None:
        """Initialize Echo model server provider.

        Args:
            model_server_url: Model server URL (defaults to ECHO_MODEL_SERVER_URL)
            model_name: Model name for encoding
            timeout: Request timeout in seconds
        """
        self.model_server_url = model_server_url or os.getenv(
            "ECHO_MODEL_SERVER_URL", "http://localhost:9000"
        )
        self.model_name = model_name
        self.timeout = timeout
        self._dimension = 768  # nomic-embed-text-v1 dimension

        logger.info(
            "echo_model_server_provider_initialized",
            extra={
                "url": self.model_server_url,
                "model": model_name,
                "dimension": self._dimension
            }
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings via Echo model server."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.model_server_url}/encoder/bi-encoder-embed",
                json={
                    "texts": texts,
                    "model_name": self.model_name,
                    "max_context_length": 512,
                    "normalize_embeddings": True,
                },
            )
            response.raise_for_status()
            data = response.json()

        embeddings = data.get("embeddings", [])

        logger.debug(
            "echo_embeddings_generated",
            extra={"count": len(texts), "model": self.model_name}
        )

        return embeddings


class EmbeddingService:
    """High-level embedding service for Vespa repositories.

    Provides a simple callable interface for embedding generation that
    can be injected into Vespa repository classes.

    Example:
        >>> service = EmbeddingService(provider=OpenAIEmbeddingProvider())
        >>> repo = VespaProjectRepository(embedding_service=service.embed)
    """

    def __init__(self, provider: EmbeddingProvider) -> None:
        """Initialize embedding service.

        Args:
            provider: Embedding provider instance
        """
        self.provider = provider

        logger.info(
            "embedding_service_initialized",
            extra={
                "provider": type(provider).__name__,
                "dimension": provider.dimension
            }
        )

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for text.

        This method can be passed directly to repository constructors.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (list of floats)
        """
        try:
            embedding = await self.provider.embed(text)
            return embedding
        except Exception as e:
            logger.exception(
                "embedding_generation_failed",
                extra={"error": str(e), "text_length": len(text)}
            )
            raise

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            embeddings = await self.provider.embed_batch(texts)
            return embeddings
        except Exception as e:
            logger.exception(
                "batch_embedding_generation_failed",
                extra={"error": str(e), "count": len(texts)}
            )
            raise

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self.provider.dimension


def create_embedding_service(
    provider_type: str = "openai",
    **kwargs,
) -> EmbeddingService:
    """Factory function to create an embedding service.

    Args:
        provider_type: Provider type ("openai", "voyage", "cohere", "echo")
        **kwargs: Additional arguments passed to provider constructor

    Returns:
        Configured EmbeddingService instance

    Raises:
        ValueError: If provider_type is unknown

    Example:
        >>> service = create_embedding_service("openai")
        >>> repo = VespaProjectRepository(embedding_service=service.embed)
    """
    providers = {
        "openai": OpenAIEmbeddingProvider,
        "voyage": VoyageEmbeddingProvider,
        "cohere": CohereEmbeddingProvider,
        "echo": EchoModelServerProvider,
    }

    if provider_type not in providers:
        raise ValueError(
            f"Unknown provider type: {provider_type}. "
            f"Supported: {list(providers.keys())}"
        )

    provider_class = providers[provider_type]
    provider = provider_class(**kwargs)

    return EmbeddingService(provider=provider)
