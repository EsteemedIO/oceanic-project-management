"""
Vespa Migration Service

Migrates project and task data from Supabase to Vespa.
Generates embeddings during migration for semantic search capabilities.

Usage:
    from server.services.vespa_migration_service import VespaMigrationService

    service = VespaMigrationService()
    result = await service.migrate_all()
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from supabase import Client

from .client_manager import get_supabase_client
from ...repositories import (
    VespaProjectRepository,
    VespaTaskRepository,
    VespaClient,
)
from ...services import EmbeddingService, OpenAIEmbeddingProvider

logger = logging.getLogger(__name__)


class MigrationStats:
    """Track migration statistics."""

    def __init__(self):
        self.projects_total = 0
        self.projects_migrated = 0
        self.projects_failed = 0
        self.tasks_total = 0
        self.tasks_migrated = 0
        self.tasks_failed = 0
        self.errors: list[dict] = []
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "projects": {
                "total": self.projects_total,
                "migrated": self.projects_migrated,
                "failed": self.projects_failed,
            },
            "tasks": {
                "total": self.tasks_total,
                "migrated": self.tasks_migrated,
                "failed": self.tasks_failed,
            },
            "errors": self.errors[:10],  # Limit errors in response
            "total_errors": len(self.errors),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": (
                (self.completed_at - self.started_at).total_seconds()
                if self.completed_at and self.started_at else None
            ),
        }


class VespaMigrationService:
    """Service for migrating data from Supabase to Vespa."""

    def __init__(
        self,
        supabase_client: Optional[Client] = None,
        vespa_client: Optional[VespaClient] = None,
        embedding_provider: Optional[Any] = None,
        org_id: str = "default",
        batch_size: int = 50,
    ):
        """Initialize migration service.

        Args:
            supabase_client: Source Supabase client
            vespa_client: Target Vespa client
            embedding_provider: Provider for generating embeddings
            org_id: Organization ID for multi-tenant filtering
            batch_size: Number of items to process per batch
        """
        self.supabase = supabase_client or get_supabase_client()
        self.vespa_client = vespa_client or VespaClient()
        self.org_id = org_id
        self.batch_size = batch_size

        # Initialize embedding provider
        if embedding_provider:
            self.embedding_provider = embedding_provider
        else:
            try:
                self.embedding_provider = OpenAIEmbeddingProvider()
            except ValueError:
                logger.warning("OpenAI API key not set, embeddings will be zero vectors")
                self.embedding_provider = None

        # Initialize repositories with shared client
        self.project_repo = VespaProjectRepository(
            client=self.vespa_client,
            embedding_service=self._embed_text if self.embedding_provider else None,
        )
        self.task_repo = VespaTaskRepository(
            client=self.vespa_client,
            embedding_service=self._embed_text if self.embedding_provider else None,
        )

        self.stats = MigrationStats()

    async def _embed_text(self, text: str) -> list[float]:
        """Generate embedding for text."""
        if self.embedding_provider:
            return await self.embedding_provider.embed(text)
        return [0.0] * 1536

    async def export_supabase_projects(self) -> list[dict]:
        """Export all projects from Supabase.

        Returns:
            List of project records from Supabase
        """
        try:
            result = self.supabase.table("archon_projects").select("*").execute()
            projects = result.data or []
            logger.info(f"Exported {len(projects)} projects from Supabase")
            return projects
        except Exception as e:
            logger.error(f"Failed to export projects from Supabase: {e}")
            raise

    async def export_supabase_tasks(self) -> list[dict]:
        """Export all tasks from Supabase.

        Returns:
            List of task records from Supabase
        """
        try:
            result = self.supabase.table("archon_tasks").select("*").execute()
            tasks = result.data or []
            logger.info(f"Exported {len(tasks)} tasks from Supabase")
            return tasks
        except Exception as e:
            logger.error(f"Failed to export tasks from Supabase: {e}")
            raise

    async def migrate_project(self, project: dict) -> bool:
        """Migrate a single project to Vespa.

        Args:
            project: Supabase project record

        Returns:
            True if successful, False otherwise
        """
        try:
            # Map Supabase fields to Vespa schema
            project_data = {
                "project_id": project.get("id"),
                "org_id": self.org_id,
                "name": project.get("title", ""),
                "description": project.get("description", "") or "",
                "status": "active",  # Default status
                "github_repo": project.get("github_repo", ""),
                "metadata": {
                    "source": "supabase_migration",
                    "original_created_at": project.get("created_at", ""),
                    "migrated_at": datetime.now(timezone.utc).isoformat(),
                },
            }

            # Extract additional metadata from data field
            data_list = project.get("data", [])
            if data_list and isinstance(data_list, list) and len(data_list) > 0:
                first_data = data_list[0]
                if isinstance(first_data, dict):
                    project_data["metadata"]["feature_number"] = first_data.get("feature_number", "")
                    project_data["metadata"]["feature_dir"] = first_data.get("feature_dir", "")

            # Create in Vespa
            result = await self.project_repo.create(project_data)

            logger.debug(f"Migrated project: {project.get('id')} - {project.get('title')}")
            return True

        except Exception as e:
            logger.error(f"Failed to migrate project {project.get('id')}: {e}")
            self.stats.errors.append({
                "type": "project",
                "id": project.get("id"),
                "title": project.get("title"),
                "error": str(e),
            })
            return False

    async def migrate_task(self, task: dict) -> bool:
        """Migrate a single task to Vespa.

        Args:
            task: Supabase task record

        Returns:
            True if successful, False otherwise
        """
        try:
            # Map Supabase fields to Vespa schema
            task_data = {
                "task_id": task.get("id"),
                "project_id": task.get("project_id", ""),
                "org_id": self.org_id,
                "title": task.get("title", ""),
                "description": task.get("description", "") or "",
                "status": task.get("status", "todo"),
                "priority": task.get("priority", "medium"),
                "assignee": task.get("assignee", ""),
                "due_date": task.get("due_date"),
                "task_order": task.get("task_order", 0),
                "metadata": {
                    "source": "supabase_migration",
                    "original_created_at": task.get("created_at", ""),
                    "migrated_at": datetime.now(timezone.utc).isoformat(),
                },
            }

            # Create in Vespa
            result = await self.task_repo.create(task_data)

            logger.debug(f"Migrated task: {task.get('id')} - {task.get('title')}")
            return True

        except Exception as e:
            logger.error(f"Failed to migrate task {task.get('id')}: {e}")
            self.stats.errors.append({
                "type": "task",
                "id": task.get("id"),
                "title": task.get("title"),
                "error": str(e),
            })
            return False

    async def migrate_projects(self, projects: list[dict]) -> int:
        """Migrate all projects to Vespa.

        Args:
            projects: List of Supabase project records

        Returns:
            Number of successfully migrated projects
        """
        self.stats.projects_total = len(projects)
        migrated = 0

        for i in range(0, len(projects), self.batch_size):
            batch = projects[i:i + self.batch_size]

            # Process batch with limited concurrency
            tasks = [self.migrate_project(p) for p in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if result is True:
                    migrated += 1
                else:
                    self.stats.projects_failed += 1

            logger.info(f"Projects: {migrated}/{self.stats.projects_total} migrated")

        self.stats.projects_migrated = migrated
        return migrated

    async def migrate_tasks(self, tasks: list[dict]) -> int:
        """Migrate all tasks to Vespa.

        Args:
            tasks: List of Supabase task records

        Returns:
            Number of successfully migrated tasks
        """
        self.stats.tasks_total = len(tasks)
        migrated = 0

        for i in range(0, len(tasks), self.batch_size):
            batch = tasks[i:i + self.batch_size]

            # Process batch with limited concurrency
            task_coroutines = [self.migrate_task(t) for t in batch]
            results = await asyncio.gather(*task_coroutines, return_exceptions=True)

            for result in results:
                if result is True:
                    migrated += 1
                else:
                    self.stats.tasks_failed += 1

            logger.info(f"Tasks: {migrated}/{self.stats.tasks_total} migrated")

        self.stats.tasks_migrated = migrated
        return migrated

    async def migrate_all(self, dry_run: bool = False) -> dict:
        """Run full migration from Supabase to Vespa.

        Args:
            dry_run: If True, only export data without importing to Vespa

        Returns:
            Migration statistics
        """
        self.stats = MigrationStats()
        self.stats.started_at = datetime.now(timezone.utc)

        logger.info("Starting Supabase → Vespa migration")

        try:
            # Export from Supabase
            logger.info("Exporting projects from Supabase...")
            projects = await self.export_supabase_projects()

            logger.info("Exporting tasks from Supabase...")
            tasks = await self.export_supabase_tasks()

            if dry_run:
                logger.info("Dry run mode - skipping Vespa import")
                self.stats.projects_total = len(projects)
                self.stats.tasks_total = len(tasks)
            else:
                # Import to Vespa
                logger.info("Migrating projects to Vespa...")
                await self.migrate_projects(projects)

                logger.info("Migrating tasks to Vespa...")
                await self.migrate_tasks(tasks)

            self.stats.completed_at = datetime.now(timezone.utc)

            logger.info(
                f"Migration completed: "
                f"{self.stats.projects_migrated}/{self.stats.projects_total} projects, "
                f"{self.stats.tasks_migrated}/{self.stats.tasks_total} tasks"
            )

            return self.stats.to_dict()

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            self.stats.completed_at = datetime.now(timezone.utc)
            self.stats.errors.append({
                "type": "migration",
                "error": str(e),
            })
            raise

    async def validate_migration(self) -> dict:
        """Validate migration by comparing counts between Supabase and Vespa.

        Returns:
            Validation results with counts from both backends
        """
        # Get Supabase counts
        supabase_projects = await self.export_supabase_projects()
        supabase_tasks = await self.export_supabase_tasks()

        # Get Vespa counts using list method with high limit
        vespa_projects = await self.project_repo.list(
            org_id=self.org_id,
            filters={"limit": 10000}
        )
        vespa_tasks = await self.task_repo.list(
            org_id=self.org_id,
            filters={"limit": 10000}
        )

        supabase_project_count = len(supabase_projects)
        supabase_task_count = len(supabase_tasks)
        vespa_project_count = len(vespa_projects) if vespa_projects else 0
        vespa_task_count = len(vespa_tasks) if vespa_tasks else 0

        return {
            "supabase": {
                "projects": supabase_project_count,
                "tasks": supabase_task_count,
            },
            "vespa": {
                "projects": vespa_project_count,
                "tasks": vespa_task_count,
            },
            "match": {
                "projects": supabase_project_count == vespa_project_count,
                "tasks": supabase_task_count == vespa_task_count,
            },
            "all_migrated": (
                supabase_project_count == vespa_project_count and
                supabase_task_count == vespa_task_count
            ),
        }


# Convenience function for running migration
async def run_migration(
    dry_run: bool = False,
    org_id: str = "default",
) -> dict:
    """Run Supabase to Vespa migration.

    Args:
        dry_run: If True, only export without importing
        org_id: Organization ID for multi-tenant filtering

    Returns:
        Migration statistics
    """
    service = VespaMigrationService(org_id=org_id)
    return await service.migrate_all(dry_run=dry_run)
