"""Vespa-backed Task Service Module

This module provides business logic for task operations using Vespa as the
backend storage instead of Supabase. Implements the same TaskService interface
for seamless integration with existing MCP tools and FastAPI endpoints.
"""

import asyncio
from datetime import datetime
from typing import Any

from src.repositories import VespaTaskRepository
from src.services import create_embedding_service

from ...config.logfire_config import get_logger

logger = get_logger(__name__)


class VespaTaskService:
    """Service class for task operations with Vespa backend"""

    VALID_STATUSES = ["todo", "doing", "review", "done"]
    VALID_PRIORITIES = ["low", "medium", "high", "critical"]

    def __init__(self, vespa_repo: VespaTaskRepository | None = None):
        """Initialize with optional VespaTaskRepository

        Args:
            vespa_repo: Optional VespaTaskRepository instance. If None, creates
                       a new instance with default embedding service.
        """
        if vespa_repo is None:
            # Create default embedding service
            embedding_service = create_embedding_service(provider_type="openai")
            self.repo = VespaTaskRepository(embedding_service=embedding_service.embed)
            logger.info("vespa_task_service_initialized", extra={"repo": "auto-created"})
        else:
            self.repo = vespa_repo
            logger.info("vespa_task_service_initialized", extra={"repo": "injected"})

    def validate_status(self, status: str) -> tuple[bool, str]:
        """Validate task status

        Args:
            status: Status to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if status not in self.VALID_STATUSES:
            return (
                False,
                f"Invalid status '{status}'. Must be one of: {', '.join(self.VALID_STATUSES)}",
            )
        return True, ""

    def validate_assignee(self, assignee: str) -> tuple[bool, str]:
        """Validate task assignee

        Args:
            assignee: Assignee to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not assignee or not isinstance(assignee, str) or len(assignee.strip()) == 0:
            return False, "Assignee must be a non-empty string"
        return True, ""

    def validate_priority(self, priority: str) -> tuple[bool, str]:
        """Validate task priority against allowed enum values

        Args:
            priority: Priority to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if priority not in self.VALID_PRIORITIES:
            return (
                False,
                f"Invalid priority '{priority}'. Must be one of: {', '.join(self.VALID_PRIORITIES)}",
            )
        return True, ""

    async def create_task(
        self,
        project_id: str,
        title: str,
        description: str = "",
        assignee: str = "User",
        task_order: int = 0,
        priority: str = "medium",
        feature: str | None = None,
        sources: list[dict[str, Any]] | None = None,
        code_examples: list[dict[str, Any]] | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Create a new task with automatic reordering.

        Args:
            project_id: Project identifier
            title: Task title
            description: Task description
            assignee: Assigned user or agent
            task_order: Order position within status column
            priority: Task priority (low, medium, high, critical)
            feature: Optional feature tag
            sources: Optional list of source documents
            code_examples: Optional list of code examples

        Returns:
            Tuple of (success, result_dict)
        """
        try:
            # Validate inputs
            if not title or not isinstance(title, str) or len(title.strip()) == 0:
                return False, {"error": "Task title is required and must be a non-empty string"}

            if not project_id or not isinstance(project_id, str):
                return False, {"error": "Project ID is required and must be a string"}

            # Validate assignee
            is_valid, error_msg = self.validate_assignee(assignee)
            if not is_valid:
                return False, {"error": error_msg}

            # Validate priority
            is_valid, error_msg = self.validate_priority(priority)
            if not is_valid:
                return False, {"error": error_msg}

            task_status = "todo"

            # REORDERING LOGIC: If inserting at a specific position, increment existing tasks
            if task_order > 0:
                # Get all tasks in the same project and status with task_order >= new task's order
                existing_tasks = await self.repo.list(
                    org_id="default",  # Vespa requires org_id for filtering
                    filters={
                        "project_id": project_id,
                        "status": task_status,
                        "limit": 1000,  # Get all tasks for reordering
                    }
                )

                # Filter tasks that need reordering
                tasks_to_reorder = [
                    task for task in existing_tasks
                    if task.get("task_order", 0) >= task_order
                ]

                if tasks_to_reorder:
                    logger.info(f"Reordering {len(tasks_to_reorder)} existing tasks")

                    # Increment task_order for all affected tasks
                    for existing_task in tasks_to_reorder:
                        new_order = existing_task["task_order"] + 1
                        await self.repo.update(
                            task_id=existing_task["task_id"],
                            updates={
                                "task_order": new_order,
                            }
                        )

            # Generate task_id (nanoid-style unique ID)
            import secrets
            task_id = f"task_{secrets.token_urlsafe(16)}"

            # Prepare task data for Vespa
            task_data = {
                "task_id": task_id,
                "org_id": "default",  # Multi-tenant support placeholder
                "project_id": project_id,
                "title": title,
                "description": description,
                "status": task_status,
                "priority": priority,
                "assignee": assignee,
                "metadata": {
                    "task_order": str(task_order),
                    "feature": feature or "",
                    "sources": str(sources or []),
                    "code_examples": str(code_examples or []),
                },
            }

            # Create task in Vespa (embedding generated automatically)
            created_task = await self.repo.create(task_data)

            return True, {
                "task": {
                    "id": created_task["task_id"],
                    "project_id": created_task["project_id"],
                    "title": created_task["title"],
                    "description": created_task["description"],
                    "status": created_task["status"],
                    "assignee": created_task["assignee"],
                    "task_order": int(created_task["metadata"].get("task_order", "0")),
                    "priority": created_task["priority"],
                    "created_at": datetime.fromtimestamp(
                        created_task["created_at"] / 1000
                    ).isoformat(),
                }
            }

        except Exception as e:
            logger.error(f"Error creating task: {e}", exc_info=True)
            return False, {"error": f"Error creating task: {str(e)}"}

    def list_tasks(
        self,
        project_id: str | None = None,
        status: str | None = None,
        include_closed: bool = False,
        exclude_large_fields: bool = False,
        include_archived: bool = False,
        search_query: str | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """List tasks with various filters.

        Args:
            project_id: Filter by project
            status: Filter by status
            include_closed: Include done tasks
            exclude_large_fields: If True, excludes sources and code_examples fields
            include_archived: If True, includes archived tasks
            search_query: Semantic search query (uses Vespa hybrid search)

        Returns:
            Tuple of (success, result_dict)
        """
        try:
            # If search query provided, use semantic search instead
            if search_query:
                return self.search_tasks(
                    query=search_query,
                    limit=100,
                    org_id="default",
                    project_id=project_id,
                )

            # Build filters for Vespa query
            filters: dict[str, Any] = {"limit": 1000}

            if project_id:
                filters["project_id"] = project_id

            if status:
                # Validate status
                is_valid, error_msg = self.validate_status(status)
                if not is_valid:
                    return False, {"error": error_msg}
                filters["status"] = status
            elif not include_closed:
                # This will need to be handled in post-processing since Vespa
                # doesn't support NOT queries easily
                pass

            # Execute async query in sync context
            loop = asyncio.get_event_loop()
            raw_tasks = loop.run_until_complete(
                self.repo.list(org_id="default", filters=filters)
            )

            # Post-process results
            tasks = []
            for task in raw_tasks:
                # Filter out done tasks if not including closed
                if not include_closed and not status and task.get("status") == "done":
                    continue

                # Filter out archived tasks if not including them
                archived = task.get("metadata", {}).get("archived", "false") == "true"
                if not include_archived and archived:
                    continue

                # Build task response
                task_data = {
                    "id": task["task_id"],
                    "project_id": task["project_id"],
                    "title": task["title"],
                    "description": task["description"],
                    "status": task["status"],
                    "assignee": task.get("assignee", "User"),
                    "task_order": int(task.get("metadata", {}).get("task_order", "0")),
                    "priority": task.get("priority", "medium"),
                    "feature": task.get("metadata", {}).get("feature"),
                    "created_at": datetime.fromtimestamp(
                        task["created_at"] / 1000
                    ).isoformat(),
                    "updated_at": datetime.fromtimestamp(
                        task["updated_at"] / 1000
                    ).isoformat(),
                    "archived": archived,
                }

                if not exclude_large_fields:
                    # Include full JSONB-like fields from metadata
                    import ast
                    try:
                        task_data["sources"] = ast.literal_eval(
                            task.get("metadata", {}).get("sources", "[]")
                        )
                        task_data["code_examples"] = ast.literal_eval(
                            task.get("metadata", {}).get("code_examples", "[]")
                        )
                    except (ValueError, SyntaxError):
                        task_data["sources"] = []
                        task_data["code_examples"] = []
                else:
                    # Add counts instead of full content
                    try:
                        import ast
                        sources = ast.literal_eval(
                            task.get("metadata", {}).get("sources", "[]")
                        )
                        code_examples = ast.literal_eval(
                            task.get("metadata", {}).get("code_examples", "[]")
                        )
                        task_data["stats"] = {
                            "sources_count": len(sources),
                            "code_examples_count": len(code_examples),
                        }
                    except (ValueError, SyntaxError):
                        task_data["stats"] = {
                            "sources_count": 0,
                            "code_examples_count": 0,
                        }

                tasks.append(task_data)

            # Sort by task_order and created_at
            tasks.sort(key=lambda t: (t["task_order"], t["created_at"]))

            filter_info = []
            if project_id:
                filter_info.append(f"project_id={project_id}")
            if status:
                filter_info.append(f"status={status}")
            if not include_closed:
                filter_info.append("excluding closed tasks")

            return True, {
                "tasks": tasks,
                "total_count": len(tasks),
                "filters_applied": ", ".join(filter_info) if filter_info else "none",
                "include_closed": include_closed,
            }

        except Exception as e:
            logger.error(f"Error listing tasks: {e}", exc_info=True)
            return False, {"error": f"Error listing tasks: {str(e)}"}

    def get_task(self, task_id: str) -> tuple[bool, dict[str, Any]]:
        """Get a specific task by ID.

        Args:
            task_id: Task unique identifier

        Returns:
            Tuple of (success, result_dict)
        """
        try:
            # Execute async query in sync context
            loop = asyncio.get_event_loop()
            task = loop.run_until_complete(self.repo.get(task_id))

            if task:
                # Convert Vespa task to standard format
                import ast
                try:
                    sources = ast.literal_eval(
                        task.get("metadata", {}).get("sources", "[]")
                    )
                    code_examples = ast.literal_eval(
                        task.get("metadata", {}).get("code_examples", "[]")
                    )
                except (ValueError, SyntaxError):
                    sources = []
                    code_examples = []

                task_data = {
                    "id": task["task_id"],
                    "project_id": task["project_id"],
                    "title": task["title"],
                    "description": task["description"],
                    "status": task["status"],
                    "assignee": task.get("assignee", "User"),
                    "task_order": int(task.get("metadata", {}).get("task_order", "0")),
                    "priority": task.get("priority", "medium"),
                    "feature": task.get("metadata", {}).get("feature"),
                    "sources": sources,
                    "code_examples": code_examples,
                    "created_at": datetime.fromtimestamp(
                        task["created_at"] / 1000
                    ).isoformat(),
                    "updated_at": datetime.fromtimestamp(
                        task["updated_at"] / 1000
                    ).isoformat(),
                    "archived": task.get("metadata", {}).get("archived", "false") == "true",
                }

                return True, {"task": task_data}
            else:
                return False, {"error": f"Task with ID {task_id} not found"}

        except Exception as e:
            logger.error(f"Error getting task: {e}", exc_info=True)
            return False, {"error": f"Error getting task: {str(e)}"}

    async def update_task(
        self, task_id: str, update_fields: dict[str, Any]
    ) -> tuple[bool, dict[str, Any]]:
        """Update task with specified fields.

        Args:
            task_id: Task identifier
            update_fields: Fields to update

        Returns:
            Tuple of (success, result_dict)
        """
        try:
            # Build update data
            updates: dict[str, Any] = {}

            # Validate and add fields
            if "title" in update_fields:
                updates["title"] = update_fields["title"]

            if "description" in update_fields:
                updates["description"] = update_fields["description"]

            if "status" in update_fields:
                is_valid, error_msg = self.validate_status(update_fields["status"])
                if not is_valid:
                    return False, {"error": error_msg}
                updates["status"] = update_fields["status"]

            if "assignee" in update_fields:
                is_valid, error_msg = self.validate_assignee(update_fields["assignee"])
                if not is_valid:
                    return False, {"error": error_msg}
                updates["assignee"] = update_fields["assignee"]

            if "priority" in update_fields:
                is_valid, error_msg = self.validate_priority(update_fields["priority"])
                if not is_valid:
                    return False, {"error": error_msg}
                updates["priority"] = update_fields["priority"]

            # Metadata fields need special handling
            if any(key in update_fields for key in ["task_order", "feature", "sources", "code_examples"]):
                # Get current task to merge metadata
                current_task = await self.repo.get(task_id)
                if not current_task:
                    return False, {"error": f"Task with ID {task_id} not found"}

                metadata = current_task.get("metadata", {}).copy()

                if "task_order" in update_fields:
                    metadata["task_order"] = str(update_fields["task_order"])

                if "feature" in update_fields:
                    metadata["feature"] = update_fields["feature"]

                if "sources" in update_fields:
                    metadata["sources"] = str(update_fields["sources"])

                if "code_examples" in update_fields:
                    metadata["code_examples"] = str(update_fields["code_examples"])

                updates["metadata"] = metadata

            # Update task in Vespa
            updated_task = await self.repo.update(task_id, updates)

            if updated_task:
                # Convert to standard format
                import ast
                try:
                    sources = ast.literal_eval(
                        updated_task.get("metadata", {}).get("sources", "[]")
                    )
                    code_examples = ast.literal_eval(
                        updated_task.get("metadata", {}).get("code_examples", "[]")
                    )
                except (ValueError, SyntaxError):
                    sources = []
                    code_examples = []

                task_data = {
                    "id": updated_task["task_id"],
                    "project_id": updated_task["project_id"],
                    "title": updated_task["title"],
                    "description": updated_task["description"],
                    "status": updated_task["status"],
                    "assignee": updated_task.get("assignee", "User"),
                    "task_order": int(updated_task.get("metadata", {}).get("task_order", "0")),
                    "priority": updated_task.get("priority", "medium"),
                    "feature": updated_task.get("metadata", {}).get("feature"),
                    "sources": sources,
                    "code_examples": code_examples,
                    "created_at": datetime.fromtimestamp(
                        updated_task["created_at"] / 1000
                    ).isoformat(),
                    "updated_at": datetime.fromtimestamp(
                        updated_task["updated_at"] / 1000
                    ).isoformat(),
                }

                return True, {"task": task_data, "message": "Task updated successfully"}
            else:
                return False, {"error": f"Task with ID {task_id} not found"}

        except Exception as e:
            logger.error(f"Error updating task: {e}", exc_info=True)
            return False, {"error": f"Error updating task: {str(e)}"}

    async def archive_task(
        self, task_id: str, archived_by: str = "mcp"
    ) -> tuple[bool, dict[str, Any]]:
        """Archive a task (soft delete).

        Args:
            task_id: Task identifier
            archived_by: User or system archiving the task

        Returns:
            Tuple of (success, result_dict)
        """
        try:
            # Check if task exists
            task = await self.repo.get(task_id)
            if not task:
                return False, {"error": f"Task with ID {task_id} not found"}

            # Check if already archived
            archived = task.get("metadata", {}).get("archived", "false") == "true"
            if archived:
                return False, {"error": f"Task with ID {task_id} is already archived"}

            # Update metadata to mark as archived
            metadata = task.get("metadata", {}).copy()
            metadata["archived"] = "true"
            metadata["archived_at"] = datetime.now().isoformat()
            metadata["archived_by"] = archived_by

            # Update task
            await self.repo.update(task_id, {"metadata": metadata})

            return True, {"task_id": task_id, "message": "Task archived successfully"}

        except Exception as e:
            logger.error(f"Error archiving task: {e}", exc_info=True)
            return False, {"error": f"Error archiving task: {str(e)}"}

    def get_all_project_task_counts(self) -> tuple[bool, dict[str, dict[str, int]]]:
        """Get task counts for all projects in a single optimized query.

        Returns task counts grouped by project_id and status.

        Returns:
            Tuple of (success, counts_dict) where counts_dict is:
            {"project-id": {"todo": 5, "doing": 2, "review": 3, "done": 10}}
        """
        try:
            logger.debug("Fetching task counts for all projects in batch")

            # Query all non-archived tasks
            loop = asyncio.get_event_loop()
            all_tasks = loop.run_until_complete(
                self.repo.list(org_id="default", filters={"limit": 10000})
            )

            # Filter out archived tasks and group by project and status
            counts_by_project: dict[str, dict[str, int]] = {}

            for task in all_tasks:
                # Skip archived tasks
                archived = task.get("metadata", {}).get("archived", "false") == "true"
                if archived:
                    continue

                project_id = task.get("project_id")
                status = task.get("status")

                if not project_id or not status:
                    continue

                # Initialize project counts if not exists
                if project_id not in counts_by_project:
                    counts_by_project[project_id] = {
                        "todo": 0,
                        "doing": 0,
                        "review": 0,
                        "done": 0,
                    }

                # Count all statuses separately
                if status in ["todo", "doing", "review", "done"]:
                    counts_by_project[project_id][status] += 1

            logger.debug(f"Task counts fetched for {len(counts_by_project)} projects")

            return True, counts_by_project

        except Exception as e:
            logger.error(f"Error fetching task counts: {e}", exc_info=True)
            return False, {"error": f"Error fetching task counts: {str(e)}"}

    def search_tasks(
        self,
        query: str,
        limit: int = 10,
        org_id: str = "default",
        workspace_id: str | None = None,
        project_id: str | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Semantic search for tasks using Vespa hybrid search.

        This is a new capability not available in the Supabase implementation.
        Uses Vespa's hybrid search combining semantic embeddings and BM25
        text matching for optimal relevance.

        Args:
            query: Search query text
            limit: Maximum results to return
            org_id: Organization identifier for filtering
            workspace_id: Optional workspace filter
            project_id: Optional project filter

        Returns:
            Tuple of (success, result_dict with tasks and relevance scores)
        """
        try:
            logger.info(
                "search_tasks_vespa_hybrid",
                extra={
                    "query": query,
                    "limit": limit,
                    "project_id": project_id,
                }
            )

            # Execute async search in sync context
            loop = asyncio.get_event_loop()
            search_results = loop.run_until_complete(
                self.repo.search(query=query, org_id=org_id, limit=limit)
            )

            # Post-filter by project_id if provided (Vespa search doesn't support filters yet)
            if project_id:
                search_results = [
                    task for task in search_results
                    if task.get("project_id") == project_id
                ]

            # Convert to standard task format
            tasks = []
            for task in search_results:
                import ast
                try:
                    sources = ast.literal_eval(
                        task.get("metadata", {}).get("sources", "[]")
                    )
                    code_examples = ast.literal_eval(
                        task.get("metadata", {}).get("code_examples", "[]")
                    )
                except (ValueError, SyntaxError):
                    sources = []
                    code_examples = []

                task_data = {
                    "id": task["task_id"],
                    "project_id": task["project_id"],
                    "title": task["title"],
                    "description": task["description"],
                    "status": task["status"],
                    "assignee": task.get("assignee", "User"),
                    "task_order": int(task.get("metadata", {}).get("task_order", "0")),
                    "priority": task.get("priority", "medium"),
                    "feature": task.get("metadata", {}).get("feature"),
                    "sources": sources,
                    "code_examples": code_examples,
                    "created_at": datetime.fromtimestamp(
                        task["created_at"] / 1000
                    ).isoformat(),
                    "updated_at": datetime.fromtimestamp(
                        task["updated_at"] / 1000
                    ).isoformat(),
                    "relevance_score": task.get("relevance", 0.0),  # Vespa relevance score
                }
                tasks.append(task_data)

            return True, {
                "tasks": tasks,
                "total_count": len(tasks),
                "query": query,
                "search_type": "hybrid_semantic_bm25",
            }

        except Exception as e:
            logger.error(f"Error searching tasks: {e}", exc_info=True)
            return False, {"error": f"Error searching tasks: {str(e)}"}
