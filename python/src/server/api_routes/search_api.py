"""
Unified Search API endpoints for Vespa-powered hybrid search

This module provides unified search endpoints that leverage Vespa's
hybrid search capabilities (semantic + BM25) to search across:
- Projects
- Tasks
- Documents (when available)

Key features:
- Semantic search using vector embeddings
- Keyword search using BM25
- Hybrid ranking combining both
- Multi-tenant filtering (org_id, workspace_id)
- Type-specific boosting
"""

from typing import Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..config.logfire_config import get_logger, logfire
from ..services.projects import (
    get_project_service,
    get_task_service,
    is_vespa_enabled,
    get_backend_info,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/api/search", tags=["search"])


class SearchRequest(BaseModel):
    """Request body for unified search"""
    query: str
    types: list[str] | None = None  # ["projects", "tasks", "documents"]
    org_id: str = "default"
    workspace_id: str | None = None
    project_id: str | None = None
    limit: int = 20
    offset: int = 0


class SearchResult(BaseModel):
    """Individual search result"""
    id: str
    type: str  # "project", "task", "document"
    title: str
    description: str | None = None
    relevance_score: float
    metadata: dict[str, Any] = {}


class SearchResponse(BaseModel):
    """Response for unified search"""
    results: list[SearchResult]
    total_count: int
    query: str
    types_searched: list[str]
    backend: str  # "vespa" or "supabase"


@router.get("/health")
async def search_health():
    """
    Health check for search API including backend status.

    Returns information about the configured storage backend
    and search capabilities.
    """
    try:
        backend_info = get_backend_info()
        vespa_enabled = is_vespa_enabled()

        return {
            "status": "healthy",
            "service": "search",
            "vespa_enabled": vespa_enabled,
            "capabilities": {
                "hybrid_search": vespa_enabled,
                "semantic_search": vespa_enabled,
                "keyword_search": True,
                "unified_search": vespa_enabled,
            },
            "backend": backend_info,
        }

    except Exception as e:
        logger.error(f"Search health check failed: {e}")
        return {
            "status": "error",
            "service": "search",
            "error": str(e),
        }


@router.post("/unified")
async def unified_search(request: SearchRequest) -> SearchResponse:
    """
    Unified search across projects, tasks, and documents.

    When Vespa backend is enabled, this performs hybrid search
    (semantic + keyword) for better results.

    When using Supabase backend, falls back to keyword-only search.

    Args:
        request: SearchRequest with query and filters

    Returns:
        SearchResponse with ranked results
    """
    try:
        logfire.info(
            f"Unified search | query={request.query} | types={request.types} | "
            f"org_id={request.org_id} | workspace_id={request.workspace_id}"
        )

        if not request.query or len(request.query.strip()) < 2:
            raise HTTPException(
                status_code=400,
                detail="Query must be at least 2 characters"
            )

        # Determine which types to search
        search_types = request.types or ["projects", "tasks"]
        results: list[SearchResult] = []
        backend = "vespa" if is_vespa_enabled() else "supabase"

        # Search projects
        if "projects" in search_types:
            project_service = get_project_service()

            if is_vespa_enabled() and hasattr(project_service, 'search_projects'):
                # Use Vespa hybrid search
                success, project_results = project_service.search_projects(
                    query=request.query,
                    limit=request.limit,
                    org_id=request.org_id,
                    workspace_id=request.workspace_id
                )

                if success:
                    for project in project_results.get("projects", []):
                        results.append(SearchResult(
                            id=project.get("id", ""),
                            type="project",
                            title=project.get("title", ""),
                            description=project.get("description", ""),
                            relevance_score=project.get("relevance_score", 0.0),
                            metadata={
                                "github_repo": project.get("github_repo", ""),
                                "created_at": project.get("created_at", ""),
                            }
                        ))
            else:
                # Fallback: Use list_projects and filter client-side
                success, project_data = project_service.list_projects(include_content=False)
                if success:
                    query_lower = request.query.lower()
                    for project in project_data.get("projects", []):
                        title = project.get("title", "").lower()
                        description = project.get("description", "").lower()
                        if query_lower in title or query_lower in description:
                            # Simple keyword match score
                            score = 1.0 if query_lower in title else 0.5
                            results.append(SearchResult(
                                id=project.get("id", ""),
                                type="project",
                                title=project.get("title", ""),
                                description=project.get("description", ""),
                                relevance_score=score,
                                metadata={
                                    "github_repo": project.get("github_repo", ""),
                                    "created_at": project.get("created_at", ""),
                                }
                            ))

        # Search tasks
        if "tasks" in search_types:
            task_service = get_task_service()

            if is_vespa_enabled() and hasattr(task_service, 'search_tasks'):
                # Use Vespa hybrid search
                success, task_results = task_service.search_tasks(
                    query=request.query,
                    limit=request.limit,
                    org_id=request.org_id,
                    workspace_id=request.workspace_id,
                    project_id=request.project_id
                )

                if success:
                    for task in task_results.get("tasks", []):
                        results.append(SearchResult(
                            id=task.get("id", ""),
                            type="task",
                            title=task.get("title", ""),
                            description=task.get("description", ""),
                            relevance_score=task.get("relevance_score", 0.0),
                            metadata={
                                "project_id": task.get("project_id", ""),
                                "status": task.get("status", ""),
                                "priority": task.get("priority", ""),
                                "assignee": task.get("assignee", ""),
                            }
                        ))
            else:
                # Fallback: Use list_tasks with search_query
                success, task_data = task_service.list_tasks(
                    project_id=request.project_id,
                    search_query=request.query,
                    include_closed=True
                )
                if success:
                    for task in task_data.get("tasks", []):
                        results.append(SearchResult(
                            id=task.get("id", ""),
                            type="task",
                            title=task.get("title", ""),
                            description=task.get("description", ""),
                            relevance_score=0.5,  # Default score for keyword search
                            metadata={
                                "project_id": task.get("project_id", ""),
                                "status": task.get("status", ""),
                                "priority": task.get("priority", ""),
                                "assignee": task.get("assignee", ""),
                            }
                        ))

        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        # Apply limit and offset
        total_count = len(results)
        results = results[request.offset:request.offset + request.limit]

        logfire.info(
            f"Search completed | query={request.query} | "
            f"total_results={total_count} | returned={len(results)} | backend={backend}"
        )

        return SearchResponse(
            results=results,
            total_count=total_count,
            query=request.query,
            types_searched=search_types,
            backend=backend
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unified search failed: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get("/projects")
async def search_projects(
    q: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(10, ge=1, le=100),
    org_id: str = Query("default"),
    workspace_id: str | None = None
):
    """
    Search projects using hybrid search (when Vespa enabled)
    or keyword search (Supabase fallback).
    """
    try:
        logfire.debug(f"Project search | query={q} | org_id={org_id}")

        project_service = get_project_service()

        if is_vespa_enabled() and hasattr(project_service, 'search_projects'):
            success, result = project_service.search_projects(
                query=q,
                limit=limit,
                org_id=org_id,
                workspace_id=workspace_id
            )

            if success:
                return {
                    "projects": result.get("projects", []),
                    "total_count": result.get("total_count", 0),
                    "query": q,
                    "backend": "vespa"
                }
            else:
                raise HTTPException(status_code=500, detail=result.get("error"))
        else:
            # Fallback to keyword search
            success, project_data = project_service.list_projects(include_content=False)
            if not success:
                raise HTTPException(status_code=500, detail=project_data.get("error"))

            query_lower = q.lower()
            matching_projects = []
            for project in project_data.get("projects", []):
                title = project.get("title", "").lower()
                description = project.get("description", "").lower()
                if query_lower in title or query_lower in description:
                    matching_projects.append(project)

            return {
                "projects": matching_projects[:limit],
                "total_count": len(matching_projects),
                "query": q,
                "backend": "supabase"
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Project search failed: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get("/tasks")
async def search_tasks(
    q: str = Query(..., min_length=2, description="Search query"),
    project_id: str | None = None,
    limit: int = Query(10, ge=1, le=100),
    org_id: str = Query("default"),
    workspace_id: str | None = None
):
    """
    Search tasks using hybrid search (when Vespa enabled)
    or keyword search (Supabase fallback).
    """
    try:
        logfire.debug(f"Task search | query={q} | project_id={project_id}")

        task_service = get_task_service()

        if is_vespa_enabled() and hasattr(task_service, 'search_tasks'):
            success, result = task_service.search_tasks(
                query=q,
                limit=limit,
                org_id=org_id,
                workspace_id=workspace_id,
                project_id=project_id
            )

            if success:
                return {
                    "tasks": result.get("tasks", []),
                    "total_count": result.get("total_count", 0),
                    "query": q,
                    "backend": "vespa"
                }
            else:
                raise HTTPException(status_code=500, detail=result.get("error"))
        else:
            # Fallback to keyword search
            success, task_data = task_service.list_tasks(
                project_id=project_id,
                search_query=q,
                include_closed=True
            )

            if not success:
                raise HTTPException(status_code=500, detail=task_data.get("error"))

            return {
                "tasks": task_data.get("tasks", [])[:limit],
                "total_count": task_data.get("total_count", 0),
                "query": q,
                "backend": "supabase"
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Task search failed: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get("/agent")
async def agent_search(
    q: str = Query(..., min_length=2, description="Natural language query"),
    context: str | None = Query(None, description="Additional context"),
    limit: int = Query(10, ge=1, le=50),
    org_id: str = Query("default"),
    workspace_id: str | None = None,
    project_id: str | None = None
):
    """
    Agent-optimized search endpoint for AI agents.

    Searches across all entity types (projects, tasks, documents)
    and returns results optimized for agent consumption.

    This endpoint is designed for:
    - MCP tool integration
    - Agentic workflows (Dolphin)
    - RAG context retrieval

    Args:
        q: Natural language search query
        context: Optional context to improve search relevance
        limit: Max results per type
        org_id: Organization filter
        workspace_id: Workspace filter
        project_id: Project filter (for task-specific searches)
    """
    try:
        full_query = f"{q} {context}" if context else q

        logfire.info(f"Agent search | query={full_query[:100]} | org_id={org_id}")

        # Use unified search
        request = SearchRequest(
            query=full_query,
            types=["projects", "tasks"],
            org_id=org_id,
            workspace_id=workspace_id,
            project_id=project_id,
            limit=limit
        )

        response = await unified_search(request)

        # Format for agent consumption
        return {
            "results": [
                {
                    "id": r.id,
                    "type": r.type,
                    "title": r.title,
                    "description": r.description,
                    "score": r.relevance_score,
                    **r.metadata
                }
                for r in response.results
            ],
            "count": response.total_count,
            "query": q,
            "search_backend": response.backend,
            "hybrid_enabled": is_vespa_enabled()
        }

    except Exception as e:
        logger.error(f"Agent search failed: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})
