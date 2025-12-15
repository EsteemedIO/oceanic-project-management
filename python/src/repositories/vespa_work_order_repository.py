"""Work order repository with Vespa backend.

Provides CRUD operations for agent work orders without embeddings.
Work orders track workflow execution state for Claude Code CLI automation.
"""

import logging
from datetime import datetime, timezone
from typing import Any, List, Optional

from .vespa_client import VespaClient

logger = logging.getLogger(__name__)


class VespaWorkOrderRepository:
    """Repository for work order CRUD operations with Vespa backend.

    Work orders track agent workflow execution state. No embedding generation
    as work orders are primarily accessed by ID or status, not semantic search.

    Schema: oceanic_work_order
    Fields:
        - work_order_id: string (document ID)
        - org_id: string (multi-tenant filtering)
        - agent_work_order_id: string (agent-specific identifier)
        - repository_url: string (git repository URL)
        - sandbox_identifier: string (sandbox/environment ID)
        - git_branch_name: string (optional - branch for work)
        - agent_session_id: string (optional - agent session)
        - status: string (pending, running, completed, failed)
        - metadata: map<string, string> (workflow metadata, error messages)
        - created_at: long (timestamp)
        - updated_at: long (timestamp)
        - completed_at: long (optional timestamp)

    Attributes:
        client: Shared VespaClient instance
        schema_name: Vespa schema name for work orders

    Example:
        >>> repo = VespaWorkOrderRepository()
        >>> work_order = await repo.create({
        ...     "work_order_id": "wo_123",
        ...     "org_id": "org_456",
        ...     "agent_work_order_id": "awo_789",
        ...     "repository_url": "https://github.com/org/repo",
        ...     "sandbox_identifier": "sandbox-123"
        ... })
    """

    def __init__(self, client: Optional[VespaClient] = None) -> None:
        """Initialize work order repository.

        Args:
            client: Shared VespaClient instance (creates new if None)
        """
        self.client = client or VespaClient()
        self.schema_name = "oceanic_work_order"
        logger.info(
            "vespa_work_order_repository_initialized",
            extra={"schema": self.schema_name}
        )

    async def create(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create a new work order.

        Sets created_at and updated_at timestamps. Status defaults to "pending".

        Args:
            data: Work order data including:
                - work_order_id: Unique identifier (required)
                - org_id: Organization identifier (required)
                - agent_work_order_id: Agent-specific ID (required)
                - repository_url: Git repository URL (required)
                - sandbox_identifier: Sandbox/environment ID (required)
                - git_branch_name: Git branch (optional)
                - agent_session_id: Agent session ID (optional)
                - status: Work order status (default: "pending")
                - metadata: Additional metadata (optional)

        Returns:
            Created work order data with all fields

        Raises:
            ValueError: If required fields missing
            httpx.HTTPStatusError: If Vespa insert fails

        Example:
            >>> work_order = await repo.create({
            ...     "work_order_id": "wo_123",
            ...     "org_id": "org_456",
            ...     "agent_work_order_id": "awo_789",
            ...     "repository_url": "https://github.com/org/repo",
            ...     "sandbox_identifier": "sandbox-123"
            ... })
        """
        # Validate required fields
        required_fields = [
            "work_order_id",
            "org_id",
            "agent_work_order_id",
            "repository_url",
            "sandbox_identifier",
        ]
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        # Prepare document fields
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        fields = {
            "work_order_id": data["work_order_id"],
            "org_id": data["org_id"],
            "agent_work_order_id": data["agent_work_order_id"],
            "repository_url": data["repository_url"],
            "sandbox_identifier": data["sandbox_identifier"],
            "git_branch_name": data.get("git_branch_name", ""),
            "agent_session_id": data.get("agent_session_id", ""),
            "status": data.get("status", "pending"),
            "metadata": data.get("metadata", {}),
            "created_at": now_ms,
            "updated_at": now_ms,
            "completed_at": 0,  # 0 means not completed
        }

        try:
            self.client.feed_data_point(
                self.schema_name,
                data["work_order_id"],
                fields,
            )

            logger.info(
                "work_order_created",
                extra={
                    "work_order_id": data["work_order_id"],
                    "org_id": data["org_id"],
                    "agent_work_order_id": data["agent_work_order_id"],
                }
            )

            return fields
        except Exception as e:
            logger.exception(
                "create_work_order_failed",
                extra={
                    "work_order_id": data.get("work_order_id"),
                    "error": str(e),
                }
            )
            raise

    async def get(self, work_order_id: str) -> Optional[dict[str, Any]]:
        """Get work order by ID.

        Args:
            work_order_id: Work order unique identifier

        Returns:
            Work order data or None if not found

        Example:
            >>> work_order = await repo.get("wo_123")
            >>> if work_order:
            ...     print(work_order["status"])
        """
        try:
            result = self.client.get_data(self.schema_name, work_order_id)

            if result:
                logger.debug(
                    "work_order_retrieved",
                    extra={"work_order_id": work_order_id}
                )
            else:
                logger.debug(
                    "work_order_not_found",
                    extra={"work_order_id": work_order_id}
                )

            return result
        except Exception as e:
            logger.exception(
                "get_work_order_failed",
                extra={"work_order_id": work_order_id, "error": str(e)}
            )
            raise

    async def list(
        self,
        org_id: str,
        filters: Optional[dict[str, Any]] = None,
    ) -> List[dict[str, Any]]:
        """List work orders for an organization with optional filters.

        Supports filtering by status, agent_work_order_id.

        Args:
            org_id: Organization identifier
            filters: Optional filters:
                - status: Filter by status (pending, running, completed, failed)
                - agent_work_order_id: Filter by agent work order ID
                - limit: Max results to return (default: 100)
                - offset: Pagination offset (default: 0)

        Returns:
            List of work order dictionaries

        Example:
            >>> work_orders = await repo.list(
            ...     "org_456",
            ...     filters={"status": "running"}
            ... )
        """
        filters = filters or {}
        limit = filters.get("limit", 100)
        offset = filters.get("offset", 0)

        # Build YQL query
        yql = f"select * from {self.schema_name} where org_id='{org_id}'"

        # Add optional filters
        if "status" in filters:
            yql += f" and status='{filters['status']}'"

        if "agent_work_order_id" in filters:
            yql += f" and agent_work_order_id='{filters['agent_work_order_id']}'"

        # Order by created_at descending
        yql += " order by created_at desc"

        params = {
            "yql": yql,
            "hits": limit,
            "offset": offset,
            "timeout": "10s",
        }

        try:
            result = self.client.query(params)
            hits = result.get("root", {}).get("children", [])

            work_orders = [hit["fields"] for hit in hits]

            logger.info(
                "work_orders_listed",
                extra={
                    "org_id": org_id,
                    "count": len(work_orders),
                    "filters": filters,
                }
            )

            return work_orders
        except Exception as e:
            logger.exception(
                "list_work_orders_failed",
                extra={"org_id": org_id, "error": str(e)}
            )
            raise

    async def update(
        self,
        work_order_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        """Partially update a work order.

        Updates updated_at timestamp automatically.
        Sets completed_at if status changes to "completed" or "failed".

        Args:
            work_order_id: Work order identifier
            updates: Fields to update (partial update)

        Returns:
            Updated work order data

        Raises:
            httpx.HTTPStatusError: If update fails

        Example:
            >>> work_order = await repo.update(
            ...     "wo_123",
            ...     {"status": "completed", "metadata": {"result": "success"}}
            ... )
        """
        # Update timestamp
        updates["updated_at"] = int(datetime.now(timezone.utc).timestamp() * 1000)

        # Set completed_at if status is terminal
        if "status" in updates and updates["status"] in ["completed", "failed"]:
            updates["completed_at"] = updates["updated_at"]

        try:
            self.client.update_data(
                self.schema_name,
                work_order_id,
                updates,
            )

            logger.info(
                "work_order_updated",
                extra={
                    "work_order_id": work_order_id,
                    "fields": list(updates.keys()),
                }
            )

            # Return updated data
            return await self.get(work_order_id) or updates
        except Exception as e:
            logger.exception(
                "update_work_order_failed",
                extra={"work_order_id": work_order_id, "error": str(e)}
            )
            raise

    async def delete(self, work_order_id: str) -> bool:
        """Delete a work order (hard delete).

        For soft delete, use update with status="deleted" or archived metadata.

        Args:
            work_order_id: Work order identifier

        Returns:
            True if deleted, False if not found

        Example:
            >>> deleted = await repo.delete("wo_123")
        """
        try:
            result = self.client.delete_data(self.schema_name, work_order_id)

            logger.info(
                "work_order_deleted",
                extra={"work_order_id": work_order_id, "deleted": result}
            )

            return result
        except Exception as e:
            logger.exception(
                "delete_work_order_failed",
                extra={"work_order_id": work_order_id, "error": str(e)}
            )
            raise
