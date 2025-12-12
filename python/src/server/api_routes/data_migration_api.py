"""Data Migration API endpoints for Enterprise Data Migration Service.

Provides REST endpoints for:
- Creating and managing data migration jobs
- Testing connections to source databases
- Schema discovery
- Field mapping configuration
- Migration execution and monitoring

Note: This is separate from migration_api.py which handles database schema migrations.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid
import logging

from ..services.migration import (
    SourceSchema,
    PostgreSQLAdapter,
    SupabaseAdapter,
)
from ..services.migration.source_adapter import SourceAdapter, ConnectionConfig
from ..services.migration.adapters.supabase_adapter import SupabaseConnectionConfig
from ..services.migration.field_mapper import (
    TableMapping,
    suggest_mappings,
)
from ..services.migration.orchestrator import (
    MigrationOrchestrator,
    MigrationConfig,
    MigrationProgress,
    MigrationStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/data-migrations", tags=["data-migrations"])

# In-memory storage for migration jobs (replace with database in production)
_migration_jobs: dict[str, MigrationOrchestrator] = {}


class ConnectionTestRequest(BaseModel):
    """Request model for testing database connection."""
    source_type: str = Field(..., description="Database type: postgresql, supabase")
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    ssl_mode: str = "prefer"
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None


class ConnectionTestResponse(BaseModel):
    """Response model for connection test."""
    success: bool
    message: str
    source_type: str
    details: Optional[dict] = None


class SchemaDiscoveryRequest(BaseModel):
    """Request model for schema discovery."""
    source_type: str
    connection: ConnectionTestRequest
    include_tables: Optional[list[str]] = None
    exclude_tables: Optional[list[str]] = None


class CreateMigrationRequest(BaseModel):
    """Request model for creating a migration job."""
    name: str
    source_type: str
    connection: ConnectionTestRequest
    org_id: str = "default"
    workspace_id: str = "default"
    include_tables: Optional[list[str]] = None
    exclude_tables: Optional[list[str]] = None
    generate_embeddings: bool = True
    batch_size: int = 1000


class MigrationJobResponse(BaseModel):
    """Response model for migration job."""
    job_id: str
    name: str
    source_type: str
    status: str
    created_at: datetime
    progress: Optional[dict] = None


class UpdateMappingsRequest(BaseModel):
    """Request model for updating field mappings."""
    table_mappings: list[TableMapping]


class StartMigrationRequest(BaseModel):
    """Request model for starting migration."""
    dry_run: bool = False


@router.post("/connect", response_model=ConnectionTestResponse)
async def test_connection(request: ConnectionTestRequest) -> ConnectionTestResponse:
    """Test connection to source database.

    Validates credentials and connectivity without starting a migration.
    """
    try:
        adapter: SourceAdapter

        if request.source_type == "postgresql":
            adapter = PostgreSQLAdapter()
            config = ConnectionConfig(
                host=request.host or "localhost",
                port=request.port or 5432,
                database=request.database or "postgres",
                username=request.username or "postgres",
                password=request.password or "",
                ssl_mode=request.ssl_mode,
            )
        elif request.source_type == "supabase":
            adapter = SupabaseAdapter()
            config = SupabaseConnectionConfig(
                host="",
                port=443,
                database="postgres",
                username="",
                password="",
                supabase_url=request.supabase_url or "",
                supabase_key=request.supabase_key or "",
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported source type: {request.source_type}"
            )

        await adapter.connect(config)
        is_healthy = await adapter.test_connection()
        await adapter.disconnect()

        return ConnectionTestResponse(
            success=is_healthy,
            message="Connection successful" if is_healthy else "Connection test failed",
            source_type=request.source_type,
            details={"host": request.host or request.supabase_url},
        )

    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return ConnectionTestResponse(
            success=False,
            message=str(e),
            source_type=request.source_type,
        )


@router.post("/schema/discover")
async def discover_schema(request: SchemaDiscoveryRequest) -> SourceSchema:
    """Discover schema from source database.

    Returns tables, columns, types, and relationships.
    """
    try:
        adapter: SourceAdapter

        if request.source_type == "postgresql":
            adapter = PostgreSQLAdapter()
            config = ConnectionConfig(
                host=request.connection.host or "localhost",
                port=request.connection.port or 5432,
                database=request.connection.database or "postgres",
                username=request.connection.username or "postgres",
                password=request.connection.password or "",
                ssl_mode=request.connection.ssl_mode,
            )
        elif request.source_type == "supabase":
            adapter = SupabaseAdapter()
            config = SupabaseConnectionConfig(
                host="",
                port=443,
                database="postgres",
                username="",
                password="",
                supabase_url=request.connection.supabase_url or "",
                supabase_key=request.connection.supabase_key or "",
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported source type: {request.source_type}"
            )

        await adapter.connect(config)
        schema = await adapter.discover_schema(
            include_tables=request.include_tables,
            exclude_tables=request.exclude_tables,
        )
        await adapter.disconnect()

        return schema

    except Exception as e:
        logger.error(f"Schema discovery failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/schema/suggest-mappings")
async def suggest_field_mappings(request: SchemaDiscoveryRequest) -> list[TableMapping]:
    """Generate AI-suggested field mappings for discovered schema.

    Uses naming conventions and types to suggest appropriate mappings.
    """
    try:
        # First discover schema
        adapter: SourceAdapter

        if request.source_type == "postgresql":
            adapter = PostgreSQLAdapter()
            config = ConnectionConfig(
                host=request.connection.host or "localhost",
                port=request.connection.port or 5432,
                database=request.connection.database or "postgres",
                username=request.connection.username or "postgres",
                password=request.connection.password or "",
                ssl_mode=request.connection.ssl_mode,
            )
        elif request.source_type == "supabase":
            adapter = SupabaseAdapter()
            config = SupabaseConnectionConfig(
                host="",
                port=443,
                database="postgres",
                username="",
                password="",
                supabase_url=request.connection.supabase_url or "",
                supabase_key=request.connection.supabase_key or "",
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported source type: {request.source_type}"
            )

        await adapter.connect(config)
        schema = await adapter.discover_schema(
            include_tables=request.include_tables,
            exclude_tables=request.exclude_tables,
        )
        await adapter.disconnect()

        # Generate mappings for each table
        mappings = []
        for table in schema.tables:
            # Determine target schema
            target_schema = _infer_target_schema(table.name)

            # Generate mapping
            mapping = suggest_mappings(table.fields, target_schema)
            mapping.source_table = table.name

            # Find primary key
            pk = next((f for f in table.fields if f.is_primary_key), None)
            if pk:
                mapping.primary_key_mapping = pk.name

            mappings.append(mapping)

        return mappings

    except Exception as e:
        logger.error(f"Mapping suggestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _infer_target_schema(table_name: str) -> str:
    """Infer Vespa target schema from table name."""
    schema_map = {
        "projects": "oceanic_project",
        "archon_projects": "oceanic_project",
        "tasks": "oceanic_task",
        "archon_tasks": "oceanic_task",
        "memories": "oceanic_memory",
        "agent_memory": "oceanic_memory",
        "documents": "oceanic_document",
    }
    return schema_map.get(table_name, f"oceanic_{table_name}")


@router.post("", response_model=MigrationJobResponse)
async def create_migration(request: CreateMigrationRequest) -> MigrationJobResponse:
    """Create a new migration job.

    The job is created in pending state and must be started separately.
    """
    job_id = str(uuid.uuid4())

    # Build connection config dict
    if request.source_type == "supabase":
        conn_dict = {
            "supabase_url": request.connection.supabase_url,
            "supabase_key": request.connection.supabase_key,
        }
    else:
        conn_dict = {
            "host": request.connection.host or "localhost",
            "port": request.connection.port or 5432,
            "database": request.connection.database or "postgres",
            "username": request.connection.username or "postgres",
            "password": request.connection.password or "",
            "ssl_mode": request.connection.ssl_mode,
        }

    config = MigrationConfig(
        job_id=job_id,
        source_type=request.source_type,
        connection_config=conn_dict,
        org_id=request.org_id,
        workspace_id=request.workspace_id,
        batch_size=request.batch_size,
        generate_embeddings=request.generate_embeddings,
        include_tables=request.include_tables,
        exclude_tables=request.exclude_tables,
    )

    orchestrator = MigrationOrchestrator(config)
    _migration_jobs[job_id] = orchestrator

    logger.info(f"Created migration job: {job_id}")

    return MigrationJobResponse(
        job_id=job_id,
        name=request.name,
        source_type=request.source_type,
        status=MigrationStatus.PENDING.value,
        created_at=datetime.utcnow(),
    )


@router.get("", response_model=list[MigrationJobResponse])
async def list_migrations() -> list[MigrationJobResponse]:
    """List all migration jobs."""
    jobs = []

    for job_id, orchestrator in _migration_jobs.items():
        jobs.append(MigrationJobResponse(
            job_id=job_id,
            name=f"Migration {job_id[:8]}",
            source_type=orchestrator.config.source_type,
            status=orchestrator.progress.status.value,
            created_at=orchestrator.progress.started_at or datetime.utcnow(),
            progress=orchestrator.progress.model_dump(),
        ))

    return jobs


@router.get("/{job_id}", response_model=MigrationJobResponse)
async def get_migration(job_id: str) -> MigrationJobResponse:
    """Get migration job details."""
    if job_id not in _migration_jobs:
        raise HTTPException(status_code=404, detail="Migration job not found")

    orchestrator = _migration_jobs[job_id]

    return MigrationJobResponse(
        job_id=job_id,
        name=f"Migration {job_id[:8]}",
        source_type=orchestrator.config.source_type,
        status=orchestrator.progress.status.value,
        created_at=orchestrator.progress.started_at or datetime.utcnow(),
        progress=orchestrator.progress.model_dump(),
    )


@router.delete("/{job_id}")
async def delete_migration(job_id: str) -> dict:
    """Delete a migration job.

    Only jobs in terminal states (completed, failed, cancelled) can be deleted.
    """
    if job_id not in _migration_jobs:
        raise HTTPException(status_code=404, detail="Migration job not found")

    orchestrator = _migration_jobs[job_id]

    if orchestrator.progress.status in (MigrationStatus.RUNNING, MigrationStatus.PAUSED):
        raise HTTPException(
            status_code=400,
            detail="Cannot delete active migration. Cancel first."
        )

    await orchestrator.cleanup()
    del _migration_jobs[job_id]

    return {"message": "Migration job deleted", "job_id": job_id}


@router.get("/{job_id}/schema")
async def get_migration_schema(job_id: str) -> SourceSchema:
    """Get discovered schema for a migration job."""
    if job_id not in _migration_jobs:
        raise HTTPException(status_code=404, detail="Migration job not found")

    orchestrator = _migration_jobs[job_id]

    # Connect if not already connected
    if not orchestrator._adapter:
        await orchestrator.connect()

    # Discover schema if not already done
    if not orchestrator._schema:
        await orchestrator.discover_schema()

    return orchestrator._schema


@router.get("/{job_id}/mappings")
async def get_migration_mappings(job_id: str) -> list[TableMapping]:
    """Get current field mappings for a migration job."""
    if job_id not in _migration_jobs:
        raise HTTPException(status_code=404, detail="Migration job not found")

    orchestrator = _migration_jobs[job_id]
    return orchestrator.config.table_mappings


@router.put("/{job_id}/mappings")
async def update_migration_mappings(
    job_id: str,
    request: UpdateMappingsRequest,
) -> dict:
    """Update field mappings for a migration job."""
    if job_id not in _migration_jobs:
        raise HTTPException(status_code=404, detail="Migration job not found")

    orchestrator = _migration_jobs[job_id]

    if orchestrator.progress.status == MigrationStatus.RUNNING:
        raise HTTPException(
            status_code=400,
            detail="Cannot update mappings while migration is running"
        )

    orchestrator.set_mappings(request.table_mappings)

    return {"message": "Mappings updated", "tables": len(request.table_mappings)}


@router.post("/{job_id}/validate")
async def validate_migration(job_id: str) -> dict:
    """Validate migration configuration before running."""
    if job_id not in _migration_jobs:
        raise HTTPException(status_code=404, detail="Migration job not found")

    orchestrator = _migration_jobs[job_id]

    # Ensure connected and schema discovered
    if not orchestrator._adapter:
        await orchestrator.connect()
    if not orchestrator._schema:
        await orchestrator.discover_schema()

    validation = await orchestrator.validate()
    return validation


@router.post("/{job_id}/dry-run")
async def dry_run_migration(job_id: str) -> MigrationProgress:
    """Run migration in dry-run mode (no actual writes)."""
    if job_id not in _migration_jobs:
        raise HTTPException(status_code=404, detail="Migration job not found")

    orchestrator = _migration_jobs[job_id]
    orchestrator.config.dry_run = True

    # Ensure connected and configured
    if not orchestrator._adapter:
        await orchestrator.connect()
    if not orchestrator._schema:
        await orchestrator.discover_schema()

    # Generate mappings if not set
    if not orchestrator.config.table_mappings:
        mappings = await orchestrator.generate_mappings()
        orchestrator.set_mappings(mappings)

    # Run dry run
    progress = await orchestrator.run()
    return progress


@router.post("/{job_id}/start")
async def start_migration(
    job_id: str,
    background_tasks: BackgroundTasks,
    request: StartMigrationRequest = StartMigrationRequest(),
) -> dict:
    """Start migration execution.

    The migration runs asynchronously. Use the progress endpoint to monitor.
    """
    if job_id not in _migration_jobs:
        raise HTTPException(status_code=404, detail="Migration job not found")

    orchestrator = _migration_jobs[job_id]

    if orchestrator.progress.status == MigrationStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Migration already running")

    orchestrator.config.dry_run = request.dry_run

    # Ensure connected and configured
    if not orchestrator._adapter:
        await orchestrator.connect()
    if not orchestrator._schema:
        await orchestrator.discover_schema()

    # Generate mappings if not set
    if not orchestrator.config.table_mappings:
        mappings = await orchestrator.generate_mappings()
        orchestrator.set_mappings(mappings)

    # Start migration in background
    async def run_migration():
        try:
            await orchestrator.run()
        except Exception as e:
            logger.error(f"Migration {job_id} failed: {e}")

    background_tasks.add_task(run_migration)

    return {
        "message": "Migration started",
        "job_id": job_id,
        "dry_run": request.dry_run,
    }


@router.post("/{job_id}/pause")
async def pause_migration(job_id: str) -> dict:
    """Pause a running migration."""
    if job_id not in _migration_jobs:
        raise HTTPException(status_code=404, detail="Migration job not found")

    orchestrator = _migration_jobs[job_id]
    await orchestrator.pause()

    return {"message": "Migration paused", "job_id": job_id}


@router.post("/{job_id}/resume")
async def resume_migration(
    job_id: str,
    background_tasks: BackgroundTasks,
) -> dict:
    """Resume a paused migration."""
    if job_id not in _migration_jobs:
        raise HTTPException(status_code=404, detail="Migration job not found")

    orchestrator = _migration_jobs[job_id]

    if orchestrator.progress.status != MigrationStatus.PAUSED:
        raise HTTPException(status_code=400, detail="Migration is not paused")

    # Resume in background
    async def resume():
        try:
            await orchestrator.resume()
        except Exception as e:
            logger.error(f"Migration {job_id} failed on resume: {e}")

    background_tasks.add_task(resume)

    return {"message": "Migration resumed", "job_id": job_id}


@router.post("/{job_id}/cancel")
async def cancel_migration(job_id: str) -> dict:
    """Cancel a running or paused migration."""
    if job_id not in _migration_jobs:
        raise HTTPException(status_code=404, detail="Migration job not found")

    orchestrator = _migration_jobs[job_id]
    await orchestrator.cancel()

    return {"message": "Migration cancelled", "job_id": job_id}


@router.get("/{job_id}/progress")
async def get_migration_progress(job_id: str) -> MigrationProgress:
    """Get current migration progress."""
    if job_id not in _migration_jobs:
        raise HTTPException(status_code=404, detail="Migration job not found")

    orchestrator = _migration_jobs[job_id]
    return orchestrator.progress
