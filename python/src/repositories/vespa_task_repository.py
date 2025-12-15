"""Task repository with Vespa backend and semantic search via embeddings.

Provides CRUD operations for tasks with automatic embedding generation
for semantic search. Tasks can be scoped to projects or standalone.
"""

import logging
from datetime import datetime, timezone
from typing import Any, List, Optional

from .vespa_client import VespaClient

logger = logging.getLogger(__name__)


class VespaTaskRepository:
    """Repository for task CRUD operations with Vespa backend.

    Tasks support both project-scoped and standalone contexts. Embeddings
    generated from title + description enable semantic search.

    Schema: oceanic_task
    Fields:
        - task_id: string (document ID)
        - org_id: string (multi-tenant filtering)
        - project_id: string (optional - null for standalone tasks)
        - title: string (indexed for text search)
        - description: string (indexed for text search)
        - status: string (todo, doing, review, done)
        - priority: string (low, medium, high, urgent)
        - assignee: string (user_id or agent_id)
        - metadata: map<string, string> (flexible storage)
        - embedding: tensor<float>(x[1536]) (semantic search)
        - created_at: long (timestamp)
        - updated_at: long (timestamp)
        - due_date: long (optional timestamp)

    Attributes:
        client: Shared VespaClient instance
        schema_name: Vespa schema name for tasks
        embedding_service: Callable for generating embeddings (injected)

    Example:
        >>> repo = VespaTaskRepository(embedding_service=generate_embedding)
        >>> task = await repo.create({
        ...     "task_id": "task_123",
        ...     "org_id": "org_456",
        ...     "project_id": "proj_789",
        ...     "title": "Implement user auth",
        ...     "description": "Add OAuth2 authentication flow"
        ... })
    """

    def __init__(
        self,
        client: Optional[VespaClient] = None,
        embedding_service: Optional[Any] = None,
    ) -> None:
        """Initialize task repository.

        Args:
            client: Shared VespaClient instance (creates new if None)
            embedding_service: Callable that takes text and returns embedding vector
                Expected signature: async def(text: str) -> List[float]
        """
        self.client = client or VespaClient()
        self.schema_name = "oceanic_task"
        self.embedding_service = embedding_service
        logger.info(
            "vespa_task_repository_initialized",
            extra={"schema": self.schema_name}
        )

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text content.

        Args:
            text: Text to embed (typically title + description)

        Returns:
            1536-dimensional embedding vector

        Raises:
            ValueError: If embedding service not configured
        """
        if self.embedding_service is None:
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
        """Create a new task with embedding generation.

        Generates embedding from title + description for semantic search.
        Sets created_at and updated_at timestamps.

        Args:
            data: Task data including:
                - task_id: Unique identifier (required)
                - org_id: Organization identifier (required)
                - project_id: Project identifier (optional)
                - title: Task title (required)
                - description: Task description (optional)
                - status: Task status (default: "todo")
                - priority: Priority level (default: "medium")
                - assignee: Assigned user/agent (optional)
                - metadata: Additional metadata (optional)
                - due_date: Due date timestamp (optional)

        Returns:
            Created task data with all fields

        Raises:
            ValueError: If required fields missing
            httpx.HTTPStatusError: If Vespa insert fails

        Example:
            >>> task = await repo.create({
            ...     "task_id": "task_123",
            ...     "org_id": "org_456",
            ...     "title": "Implement auth",
            ...     "description": "OAuth2 flow"
            ... })
        """
        # Validate required fields
        required_fields = ["task_id", "org_id", "title"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        # Generate embedding from title + description
        embed_text = f"{data['title']} {data.get('description', '')}"
        embedding = await self._generate_embedding(embed_text)

        # Prepare document fields
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        fields = {
            "task_id": data["task_id"],
            "org_id": data["org_id"],
            "project_id": data.get("project_id", ""),  # Empty string for standalone
            "title": data["title"],
            "description": data.get("description", ""),
            "status": data.get("status", "todo"),
            "priority": data.get("priority", "medium"),
            "assignee": data.get("assignee", ""),
            "metadata": data.get("metadata", {}),
            "embedding": {"values": embedding},
            "created_at": now_ms,
            "updated_at": now_ms,
            "due_date": data.get("due_date", 0),  # 0 means no due date
        }

        try:
            self.client.feed_data_point(
                self.schema_name,
                data["task_id"],
                fields,
            )

            logger.info(
                "task_created",
                extra={
                    "task_id": data["task_id"],
                    "org_id": data["org_id"],
                    "project_id": data.get("project_id"),
                    "title": data["title"],
                }
            )

            return fields
        except Exception as e:
            logger.exception(
                "create_task_failed",
                extra={
                    "task_id": data.get("task_id"),
                    "error": str(e),
                }
            )
            raise

    async def get(self, task_id: str) -> Optional[dict[str, Any]]:
        """Get task by ID.

        Args:
            task_id: Task unique identifier

        Returns:
            Task data or None if not found

        Example:
            >>> task = await repo.get("task_123")
            >>> if task:
            ...     print(task["title"])
        """
        try:
            result = self.client.get_data(self.schema_name, task_id)

            if result:
                logger.debug(
                    "task_retrieved",
                    extra={"task_id": task_id}
                )
            else:
                logger.debug(
                    "task_not_found",
                    extra={"task_id": task_id}
                )

            return result
        except Exception as e:
            logger.exception(
                "get_task_failed",
                extra={"task_id": task_id, "error": str(e)}
            )
            raise

    async def list(
        self,
        org_id: str,
        filters: Optional[dict[str, Any]] = None,
    ) -> List[dict[str, Any]]:
        """List tasks for an organization with optional filters.

        Supports filtering by project_id, status, priority, assignee.

        Args:
            org_id: Organization identifier
            filters: Optional filters:
                - project_id: Filter by project (include null for standalone)
                - status: Filter by status (todo, doing, review, done)
                - priority: Filter by priority (low, medium, high, urgent)
                - assignee: Filter by assigned user/agent
                - limit: Max results to return (default: 100)
                - offset: Pagination offset (default: 0)

        Returns:
            List of task dictionaries

        Example:
            >>> tasks = await repo.list(
            ...     "org_456",
            ...     filters={"project_id": "proj_789", "status": "doing"}
            ... )
        """
        filters = filters or {}
        limit = filters.get("limit", 100)
        offset = filters.get("offset", 0)

        # Build YQL query
        yql = f"select * from {self.schema_name} where org_id='{org_id}'"

        # Add optional filters
        if "project_id" in filters:
            yql += f" and project_id='{filters['project_id']}'"

        if "status" in filters:
            yql += f" and status='{filters['status']}'"

        if "priority" in filters:
            yql += f" and priority='{filters['priority']}'"

        if "assignee" in filters:
            yql += f" and assignee='{filters['assignee']}'"

        # Order by updated_at descending
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

            tasks = [hit["fields"] for hit in hits]

            logger.info(
                "tasks_listed",
                extra={
                    "org_id": org_id,
                    "count": len(tasks),
                    "filters": filters,
                }
            )

            return tasks
        except Exception as e:
            logger.exception(
                "list_tasks_failed",
                extra={"org_id": org_id, "error": str(e)}
            )
            raise

    async def update(
        self,
        task_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        """Partially update a task.

        Regenerates embedding if title or description changed.
        Updates updated_at timestamp automatically.

        Args:
            task_id: Task identifier
            updates: Fields to update (partial update)

        Returns:
            Updated task data

        Raises:
            httpx.HTTPStatusError: If update fails

        Example:
            >>> task = await repo.update(
            ...     "task_123",
            ...     {"status": "done", "assignee": "user_456"}
            ... )
        """
        # Check if we need to regenerate embedding
        needs_embedding = "title" in updates or "description" in updates

        if needs_embedding:
            # Fetch current data to merge with updates
            current = await self.get(task_id)
            if not current:
                raise ValueError(f"Task not found: {task_id}")

            # Merge updates to get full text for embedding
            title = updates.get("title", current.get("title", ""))
            description = updates.get("description", current.get("description", ""))
            embed_text = f"{title} {description}"

            embedding = await self._generate_embedding(embed_text)
            updates["embedding"] = {"values": embedding}

        # Update timestamp
        updates["updated_at"] = int(datetime.now(timezone.utc).timestamp() * 1000)

        try:
            self.client.update_data(
                self.schema_name,
                task_id,
                updates,
            )

            logger.info(
                "task_updated",
                extra={
                    "task_id": task_id,
                    "fields": list(updates.keys()),
                    "embedding_regenerated": needs_embedding,
                }
            )

            # Return updated data
            return await self.get(task_id) or updates
        except Exception as e:
            logger.exception(
                "update_task_failed",
                extra={"task_id": task_id, "error": str(e)}
            )
            raise

    async def delete(self, task_id: str) -> bool:
        """Delete a task (hard delete).

        For soft delete, use update with status="deleted" or archived metadata.

        Args:
            task_id: Task identifier

        Returns:
            True if deleted, False if not found

        Example:
            >>> deleted = await repo.delete("task_123")
        """
        try:
            result = self.client.delete_data(self.schema_name, task_id)

            logger.info(
                "task_deleted",
                extra={"task_id": task_id, "deleted": result}
            )

            return result
        except Exception as e:
            logger.exception(
                "delete_task_failed",
                extra={"task_id": task_id, "error": str(e)}
            )
            raise

    async def search(
        self,
        query: str,
        org_id: str,
        limit: int = 10,
    ) -> List[dict[str, Any]]:
        """Semantic search for tasks using embeddings.

        Generates embedding for query and performs nearest neighbor search
        combined with text matching on title/description fields.

        Args:
            query: Search query text
            org_id: Organization identifier (scopes results)
            limit: Maximum results to return (default: 10)

        Returns:
            List of tasks ranked by relevance

        Example:
            >>> results = await repo.search(
            ...     "authentication OAuth",
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

            tasks = [hit["fields"] for hit in hits]

            logger.info(
                "tasks_searched",
                extra={
                    "query": query,
                    "org_id": org_id,
                    "results": len(tasks),
                }
            )

            return tasks
        except Exception as e:
            logger.exception(
                "search_tasks_failed",
                extra={"query": query, "org_id": org_id, "error": str(e)}
            )
            raise
