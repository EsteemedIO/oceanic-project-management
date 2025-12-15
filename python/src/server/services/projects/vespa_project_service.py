"""
Vespa-backed Project Service Module for Archon

This module provides the Vespa implementation of project operations,
matching the interface of the Supabase-backed ProjectService.
Uses the VespaProjectRepository for hybrid search and vector embeddings.
"""

import os
from datetime import datetime
from typing import Any
import asyncio

from src.repositories import VespaProjectRepository
from src.services import create_embedding_service
from ...config.logfire_config import get_logger

logger = get_logger(__name__)

# Singleton instances for reuse
_project_repo: VespaProjectRepository | None = None
_embedding_service = None


def get_project_repo() -> VespaProjectRepository:
    """Get or create the Vespa project repository singleton."""
    global _project_repo, _embedding_service

    if _project_repo is None:
        # Determine embedding provider from env
        provider = os.getenv("EMBEDDING_PROVIDER", "openai")
        _embedding_service = create_embedding_service(provider)
        _project_repo = VespaProjectRepository(embedding_service=_embedding_service)

    return _project_repo


class VespaProjectService:
    """Service class for project operations using Vespa backend"""

    def __init__(self, vespa_repo: VespaProjectRepository | None = None):
        """Initialize with optional Vespa repository"""
        self.repo = vespa_repo or get_project_repo()

    def create_project(
        self,
        title: str,
        github_repo: str = None,
        description: str = None,
        org_id: str = "default",
        workspace_id: str = "default",
        **kwargs
    ) -> tuple[bool, dict[str, Any]]:
        """
        Create a new project in Vespa with optional fields.

        Returns:
            Tuple of (success, result_dict)
        """
        try:
            # Validate inputs
            if not title or not isinstance(title, str) or len(title.strip()) == 0:
                return False, {"error": "Project title is required and must be a non-empty string"}

            # Build project data
            project_data = {
                "title": title.strip(),
                "description": description or "",
                "github_repo": github_repo.strip() if github_repo else "",
                "org_id": org_id,
                "workspace_id": workspace_id,
                "docs": kwargs.get("docs", []),
                "features": kwargs.get("features", []),
                "data": kwargs.get("data", []),
                "pinned": kwargs.get("pinned", False),
            }

            # Run async operation in sync context
            result = asyncio.get_event_loop().run_until_complete(
                self.repo.create(project_data)
            )

            project_id = result.get("project_id")
            logger.info(f"Vespa project created successfully with ID: {project_id}")

            return True, {
                "project_id": project_id,
                "project": {
                    "id": project_id,
                    "title": project_data["title"],
                    "github_repo": project_data.get("github_repo"),
                    "description": project_data.get("description", ""),
                    "created_at": result.get("created_at"),
                    "docs": project_data.get("docs", []),
                    "features": project_data.get("features", []),
                    "data": project_data.get("data", []),
                    "pinned": project_data.get("pinned", False),
                }
            }

        except Exception as e:
            logger.error(f"Error creating Vespa project: {e}")
            return False, {"error": f"Database error: {str(e)}"}

    def list_projects(self, include_content: bool = True) -> tuple[bool, dict[str, Any]]:
        """
        List all projects from Vespa.

        Args:
            include_content: If True (default), includes docs, features, data fields.
                           If False, returns lightweight metadata only with counts.

        Returns:
            Tuple of (success, result_dict)
        """
        try:
            # Run async operation in sync context
            result = asyncio.get_event_loop().run_until_complete(
                self.repo.list_all()
            )

            vespa_projects = result.get("projects", [])

            projects = []
            for project in vespa_projects:
                if include_content:
                    projects.append({
                        "id": project.get("project_id"),
                        "title": project.get("title", ""),
                        "github_repo": project.get("github_repo", ""),
                        "created_at": project.get("created_at", ""),
                        "updated_at": project.get("updated_at", ""),
                        "pinned": project.get("pinned", False),
                        "description": project.get("description", ""),
                        "docs": project.get("docs", []),
                        "features": project.get("features", []),
                        "data": project.get("data", []),
                    })
                else:
                    # Lightweight response - metadata + stats only
                    docs = project.get("docs", [])
                    features = project.get("features", [])
                    data = project.get("data", [])

                    projects.append({
                        "id": project.get("project_id"),
                        "title": project.get("title", ""),
                        "github_repo": project.get("github_repo", ""),
                        "created_at": project.get("created_at", ""),
                        "updated_at": project.get("updated_at", ""),
                        "pinned": project.get("pinned", False),
                        "description": project.get("description", ""),
                        "stats": {
                            "docs_count": len(docs) if isinstance(docs, list) else 0,
                            "features_count": len(features) if isinstance(features, list) else 0,
                            "has_data": bool(data)
                        }
                    })

            return True, {"projects": projects, "total_count": len(projects)}

        except Exception as e:
            logger.error(f"Error listing Vespa projects: {e}")
            return False, {"error": f"Error listing projects: {str(e)}"}

    def get_project(self, project_id: str) -> tuple[bool, dict[str, Any]]:
        """
        Get a specific project by ID from Vespa.

        Returns:
            Tuple of (success, result_dict)
        """
        try:
            # Run async operation in sync context
            result = asyncio.get_event_loop().run_until_complete(
                self.repo.get(project_id)
            )

            if result.get("found"):
                project = result.get("project", {})

                # Transform to expected format
                formatted_project = {
                    "id": project.get("project_id"),
                    "title": project.get("title", ""),
                    "github_repo": project.get("github_repo", ""),
                    "description": project.get("description", ""),
                    "created_at": project.get("created_at", ""),
                    "updated_at": project.get("updated_at", ""),
                    "pinned": project.get("pinned", False),
                    "docs": project.get("docs", []),
                    "features": project.get("features", []),
                    "data": project.get("data", []),
                    # Empty sources for now - can be extended
                    "technical_sources": [],
                    "business_sources": [],
                }

                return True, {"project": formatted_project}
            else:
                return False, {"error": f"Project with ID {project_id} not found"}

        except Exception as e:
            logger.error(f"Error getting Vespa project: {e}")
            return False, {"error": f"Error getting project: {str(e)}"}

    def delete_project(self, project_id: str) -> tuple[bool, dict[str, Any]]:
        """
        Delete a project from Vespa.

        Note: Unlike Supabase, Vespa doesn't have cascading deletes,
        so tasks must be deleted separately.

        Returns:
            Tuple of (success, result_dict)
        """
        try:
            # First check if project exists
            get_result = asyncio.get_event_loop().run_until_complete(
                self.repo.get(project_id)
            )

            if not get_result.get("found"):
                return False, {"error": f"Project with ID {project_id} not found"}

            # Delete the project
            result = asyncio.get_event_loop().run_until_complete(
                self.repo.delete(project_id)
            )

            # TODO: Also delete associated tasks from Vespa
            # For now, return success with 0 deleted tasks
            return True, {
                "project_id": project_id,
                "deleted_tasks": 0,  # Tasks deletion not implemented yet
                "message": "Project deleted successfully",
            }

        except Exception as e:
            logger.error(f"Error deleting Vespa project: {e}")
            return False, {"error": f"Error deleting project: {str(e)}"}

    def get_project_features(self, project_id: str) -> tuple[bool, dict[str, Any]]:
        """
        Get features from a project's features field in Vespa.

        Returns:
            Tuple of (success, result_dict)
        """
        try:
            # Get the project first
            result = asyncio.get_event_loop().run_until_complete(
                self.repo.get(project_id)
            )

            if not result.get("found"):
                return False, {"error": "Project not found"}

            project = result.get("project", {})
            features = project.get("features", [])

            # Extract feature labels for dropdown options
            feature_options = []
            for feature in features:
                if isinstance(feature, dict) and "data" in feature and "label" in feature["data"]:
                    feature_options.append({
                        "id": feature.get("id", ""),
                        "label": feature["data"]["label"],
                        "type": feature["data"].get("type", ""),
                        "feature_type": feature.get("type", "page"),
                    })

            return True, {"features": feature_options, "count": len(feature_options)}

        except Exception as e:
            logger.error(f"Error getting Vespa project features: {e}")
            return False, {"error": f"Error getting project features: {str(e)}"}

    def update_project(
        self, project_id: str, update_fields: dict[str, Any]
    ) -> tuple[bool, dict[str, Any]]:
        """
        Update a project in Vespa with specified fields.

        Returns:
            Tuple of (success, result_dict)
        """
        try:
            # Build update data
            update_data = {"updated_at": datetime.now().isoformat()}

            # Add allowed fields
            allowed_fields = [
                "title",
                "description",
                "github_repo",
                "docs",
                "features",
                "data",
                "pinned",
            ]

            for field in allowed_fields:
                if field in update_fields:
                    update_data[field] = update_fields[field]

            # Handle pinning logic - only one project can be pinned at a time
            if update_fields.get("pinned") is True:
                # In Vespa, we'd need to query and update all pinned projects
                # For now, just update this one - full implementation would need batch update
                logger.debug(f"Pinning project {project_id}")

            # Update the project
            result = asyncio.get_event_loop().run_until_complete(
                self.repo.update(project_id, update_data)
            )

            if result.get("success"):
                # Fetch updated project
                get_result = asyncio.get_event_loop().run_until_complete(
                    self.repo.get(project_id)
                )

                if get_result.get("found"):
                    project = get_result.get("project", {})
                    formatted_project = {
                        "id": project.get("project_id"),
                        "title": project.get("title", ""),
                        "github_repo": project.get("github_repo", ""),
                        "description": project.get("description", ""),
                        "created_at": project.get("created_at", ""),
                        "updated_at": project.get("updated_at", ""),
                        "pinned": project.get("pinned", False),
                        "docs": project.get("docs", []),
                        "features": project.get("features", []),
                        "data": project.get("data", []),
                    }
                    return True, {"project": formatted_project, "message": "Project updated successfully"}

            return False, {"error": f"Project with ID {project_id} not found"}

        except Exception as e:
            logger.error(f"Error updating Vespa project: {e}")
            return False, {"error": f"Error updating project: {str(e)}"}

    def search_projects(
        self,
        query: str,
        limit: int = 10,
        org_id: str = "default",
        workspace_id: str | None = None
    ) -> tuple[bool, dict[str, Any]]:
        """
        Search projects using Vespa hybrid search (semantic + keyword).

        This is a new capability not available in the Supabase implementation.

        Args:
            query: Search query string
            limit: Maximum number of results
            org_id: Organization ID for multi-tenant filtering
            workspace_id: Optional workspace ID for filtering

        Returns:
            Tuple of (success, result_dict)
        """
        try:
            # Build filters
            filters = [f'org_id contains "{org_id}"']
            if workspace_id:
                filters.append(f'workspace_id contains "{workspace_id}"')

            # Run async search
            result = asyncio.get_event_loop().run_until_complete(
                self.repo.hybrid_search(
                    query=query,
                    limit=limit,
                    filters=filters
                )
            )

            projects = result.get("results", [])

            # Format results
            formatted_projects = []
            for project in projects:
                formatted_projects.append({
                    "id": project.get("project_id"),
                    "title": project.get("title", ""),
                    "description": project.get("description", ""),
                    "github_repo": project.get("github_repo", ""),
                    "relevance_score": project.get("relevance"),
                    "created_at": project.get("created_at", ""),
                })

            return True, {
                "projects": formatted_projects,
                "total_count": len(formatted_projects),
                "query": query
            }

        except Exception as e:
            logger.error(f"Error searching Vespa projects: {e}")
            return False, {"error": f"Error searching projects: {str(e)}"}
