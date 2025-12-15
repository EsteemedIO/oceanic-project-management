"""Project repository with Vespa backend and semantic search via embeddings.

Provides CRUD operations for projects with automatic embedding generation
for semantic search capabilities. Replaces Supabase project storage.
"""

import logging
from datetime import datetime, timezone
from typing import Any, List, Optional

from .vespa_client import VespaClient

logger = logging.getLogger(__name__)


class VespaProjectRepository:
    """Repository for project CRUD operations with Vespa backend.

    Projects are stored in Vespa with embeddings generated from name + description
    for semantic search. Each project belongs to an organization (org_id).

    Schema: oceanic_project
    Fields:
        - project_id: string (document ID)
        - org_id: string (multi-tenant filtering)
        - name: string (indexed for text search)
        - description: string (indexed for text search)
        - status: string (active, archived, deleted)
        - metadata: map<string, string> (flexible JSONB-like storage)
        - embedding: tensor<float>(x[1536]) (semantic search)
        - created_at: long (timestamp)
        - updated_at: long (timestamp)

    Attributes:
        client: Shared VespaClient instance
        schema_name: Vespa schema name for projects
        embedding_service: Callable for generating embeddings (injected)

    Example:
        >>> repo = VespaProjectRepository(embedding_service=generate_embedding)
        >>> project = await repo.create({
        ...     "project_id": "proj_123",
        ...     "org_id": "org_456",
        ...     "name": "Q4 Roadmap",
        ...     "description": "Platform improvements for Q4 2025"
        ... })
    """

    def __init__(
        self,
        client: Optional[VespaClient] = None,
        embedding_service: Optional[Any] = None,
    ) -> None:
        """Initialize project repository.

        Args:
            client: Shared VespaClient instance (creates new if None)
            embedding_service: Callable that takes text and returns embedding vector
                Expected signature: async def(text: str) -> List[float]
        """
        self.client = client or VespaClient()
        self.schema_name = "oceanic_project"
        self.embedding_service = embedding_service
        logger.info(
            "vespa_project_repository_initialized",
            extra={"schema": self.schema_name}
        )

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text content.

        Args:
            text: Text to embed (typically name + description)

        Returns:
            1536-dimensional embedding vector

        Raises:
            ValueError: If embedding service not configured
        """
        if self.embedding_service is None:
            # Return zero vector as placeholder if no embedding service
            logger.warning(
                "embedding_service_not_configured",
                extra={"using_placeholder": True}
            )
            return [0.0] * 1536

        try:
            embedding = await self.embedding_service(text)
            logger.debug(
                "embedding_generated",
                extra={"text_length": len(text), "embedding_dim": len(embedding)}
            )
            return embedding
        except Exception as e:
            logger.exception(
                "embedding_generation_failed",
                extra={"error": str(e)}
            )
            raise

    async def create(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create a new project with embedding generation.

        Generates embedding from name + description for semantic search.
        Sets created_at and updated_at timestamps.

        Args:
            data: Project data including:
                - project_id: Unique identifier (required)
                - org_id: Organization identifier (required)
                - name: Project name (required)
                - description: Project description (optional)
                - status: Project status (default: "active")
                - metadata: Additional metadata (optional)

        Returns:
            Created project data with all fields

        Raises:
            ValueError: If required fields missing
            httpx.HTTPStatusError: If Vespa insert fails

        Example:
            >>> project = await repo.create({
            ...     "project_id": "proj_123",
            ...     "org_id": "org_456",
            ...     "name": "Q4 Roadmap",
            ...     "description": "Platform improvements"
            ... })
        """
        # Validate required fields
        required_fields = ["project_id", "org_id", "name"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        # Generate embedding from name + description
        embed_text = f"{data['name']} {data.get('description', '')}"
        embedding = await self._generate_embedding(embed_text)

        # Prepare document fields
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        fields = {
            "project_id": data["project_id"],
            "org_id": data["org_id"],
            "name": data["name"],
            "description": data.get("description", ""),
            "status": data.get("status", "active"),
            "metadata": data.get("metadata", {}),
            "embedding": {"values": embedding},  # Vespa tensor format
            "created_at": now_ms,
            "updated_at": now_ms,
        }

        try:
            self.client.feed_data_point(
                self.schema_name,
                data["project_id"],
                fields,
            )

            logger.info(
                "project_created",
                extra={
                    "project_id": data["project_id"],
                    "org_id": data["org_id"],
                    "name": data["name"],
                }
            )

            return fields
        except Exception as e:
            logger.exception(
                "create_project_failed",
                extra={
                    "project_id": data.get("project_id"),
                    "error": str(e),
                }
            )
            raise

    async def get(self, project_id: str) -> Optional[dict[str, Any]]:
        """Get project by ID.

        Args:
            project_id: Project unique identifier

        Returns:
            Project data or None if not found

        Example:
            >>> project = await repo.get("proj_123")
            >>> if project:
            ...     print(project["name"])
        """
        try:
            result = self.client.get_data(self.schema_name, project_id)

            if result:
                logger.debug(
                    "project_retrieved",
                    extra={"project_id": project_id}
                )
            else:
                logger.debug(
                    "project_not_found",
                    extra={"project_id": project_id}
                )

            return result
        except Exception as e:
            logger.exception(
                "get_project_failed",
                extra={"project_id": project_id, "error": str(e)}
            )
            raise

    async def list(
        self,
        org_id: str,
        filters: Optional[dict[str, Any]] = None,
    ) -> List[dict[str, Any]]:
        """List projects for an organization with optional filters.

        Uses YQL query to filter by org_id and optional status/metadata.

        Args:
            org_id: Organization identifier
            filters: Optional filters:
                - status: Filter by status (e.g., "active", "archived")
                - limit: Max results to return (default: 100)
                - offset: Pagination offset (default: 0)

        Returns:
            List of project dictionaries

        Example:
            >>> projects = await repo.list(
            ...     "org_456",
            ...     filters={"status": "active", "limit": 10}
            ... )
        """
        filters = filters or {}
        limit = filters.get("limit", 100)
        offset = filters.get("offset", 0)

        # Build YQL query
        yql = f"select * from {self.schema_name} where org_id='{org_id}'"

        # Add status filter if provided
        if "status" in filters:
            yql += f" and status='{filters['status']}'"

        # Add ordering by updated_at descending
        yql += " order by updated_at desc"

        params = {
            "yql": yql,
            "hits": limit,
            "offset": offset,
            "timeout": "10s",
        }

        try:
            result = self.client.query(params)
            hits = result.get("root", {}).get("children", [])

            projects = [hit["fields"] for hit in hits]

            logger.info(
                "projects_listed",
                extra={
                    "org_id": org_id,
                    "count": len(projects),
                    "filters": filters,
                }
            )

            return projects
        except Exception as e:
            logger.exception(
                "list_projects_failed",
                extra={"org_id": org_id, "error": str(e)}
            )
            raise

    async def update(
        self,
        project_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        """Partially update a project.

        Regenerates embedding if name or description changed.
        Updates updated_at timestamp automatically.

        Args:
            project_id: Project identifier
            updates: Fields to update (partial update)

        Returns:
            Updated project data

        Raises:
            httpx.HTTPStatusError: If update fails

        Example:
            >>> project = await repo.update(
            ...     "proj_123",
            ...     {"name": "Updated Roadmap", "status": "archived"}
            ... )
        """
        # Check if we need to regenerate embedding
        needs_embedding = "name" in updates or "description" in updates

        if needs_embedding:
            # Fetch current data to merge with updates
            current = await self.get(project_id)
            if not current:
                raise ValueError(f"Project not found: {project_id}")

            # Merge updates to get full text for embedding
            name = updates.get("name", current.get("name", ""))
            description = updates.get("description", current.get("description", ""))
            embed_text = f"{name} {description}"

            embedding = await self._generate_embedding(embed_text)
            updates["embedding"] = {"values": embedding}

        # Update timestamp
        updates["updated_at"] = int(datetime.now(timezone.utc).timestamp() * 1000)

        try:
            self.client.update_data(
                self.schema_name,
                project_id,
                updates,
            )

            logger.info(
                "project_updated",
                extra={
                    "project_id": project_id,
                    "fields": list(updates.keys()),
                    "embedding_regenerated": needs_embedding,
                }
            )

            # Return updated data
            return await self.get(project_id) or updates
        except Exception as e:
            logger.exception(
                "update_project_failed",
                extra={"project_id": project_id, "error": str(e)}
            )
            raise

    async def delete(self, project_id: str) -> bool:
        """Delete a project (hard delete).

        For soft delete, use update with status="deleted".

        Args:
            project_id: Project identifier

        Returns:
            True if deleted, False if not found

        Example:
            >>> deleted = await repo.delete("proj_123")
        """
        try:
            result = self.client.delete_data(self.schema_name, project_id)

            logger.info(
                "project_deleted",
                extra={"project_id": project_id, "deleted": result}
            )

            return result
        except Exception as e:
            logger.exception(
                "delete_project_failed",
                extra={"project_id": project_id, "error": str(e)}
            )
            raise

    async def search(
        self,
        query: str,
        org_id: str,
        limit: int = 10,
    ) -> List[dict[str, Any]]:
        """Semantic search for projects using embeddings.

        Generates embedding for query and performs nearest neighbor search
        combined with text matching on name/description fields.

        Args:
            query: Search query text
            org_id: Organization identifier (scopes results)
            limit: Maximum results to return (default: 10)

        Returns:
            List of projects ranked by relevance

        Example:
            >>> results = await repo.search(
            ...     "platform improvements Q4",
            ...     "org_456",
            ...     limit=5
            ... )
        """
        # Generate query embedding
        query_embedding = await self._generate_embedding(query)

        # Build hybrid search YQL (semantic + text)
        yql = (
            f"select * from {self.schema_name} where "
            f"org_id='{org_id}' and "
            f"({{targetHits: {limit * 10}}}nearestNeighbor(embedding, query_embedding) "
            f"or userQuery())"
        )

        params = {
            "yql": yql,
            "query": query,  # For text matching
            "input.query(query_embedding)": str(query_embedding),
            "hits": limit,
            "ranking.profile": "hybrid_search",  # Assumes schema defines this
            "timeout": "10s",
        }

        try:
            result = self.client.query(params)
            hits = result.get("root", {}).get("children", [])

            projects = [hit["fields"] for hit in hits]

            logger.info(
                "projects_searched",
                extra={
                    "query": query,
                    "org_id": org_id,
                    "results": len(projects),
                }
            )

            return projects
        except Exception as e:
            logger.exception(
                "search_projects_failed",
                extra={"query": query, "org_id": org_id, "error": str(e)}
            )
            raise
