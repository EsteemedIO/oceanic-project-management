"""
Vespa Migration API endpoints.

Provides REST endpoints for migrating data from Supabase to Vespa.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..config.logfire_config import get_logger
from ..services.vespa_migration_service import VespaMigrationService

logger = get_logger(__name__)

router = APIRouter(prefix="/api/migration/vespa", tags=["vespa-migration"])


class MigrationResponse(BaseModel):
    """Response model for migration status."""
    success: bool
    message: str
    stats: dict | None = None


class ValidationResponse(BaseModel):
    """Response model for validation results."""
    supabase: dict
    vespa: dict
    match: dict
    all_migrated: bool


@router.get("/status")
async def migration_status():
    """Get current migration status and backend info.

    Returns information about what data exists in Supabase
    and what has been migrated to Vespa.
    """
    try:
        service = VespaMigrationService()
        validation = await service.validate_migration()

        return {
            "status": "ready",
            "backends": {
                "source": "supabase",
                "target": "vespa",
            },
            "data": validation,
        }

    except Exception as e:
        logger.error(f"Failed to get migration status: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


@router.post("/dry-run")
async def migration_dry_run(
    org_id: str = Query("default", description="Organization ID"),
) -> MigrationResponse:
    """Run migration in dry-run mode (export only, no import).

    This will export all data from Supabase without writing to Vespa,
    useful for previewing what will be migrated.
    """
    try:
        logger.info(f"Starting dry-run migration for org_id={org_id}")

        service = VespaMigrationService(org_id=org_id)
        stats = await service.migrate_all(dry_run=True)

        return MigrationResponse(
            success=True,
            message=f"Dry run completed: {stats['projects']['total']} projects, {stats['tasks']['total']} tasks would be migrated",
            stats=stats,
        )

    except Exception as e:
        logger.error(f"Dry-run migration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run")
async def run_migration(
    org_id: str = Query("default", description="Organization ID"),
) -> MigrationResponse:
    """Run full migration from Supabase to Vespa.

    WARNING: This will copy all projects and tasks to Vespa.
    Existing data in Vespa with the same IDs will be overwritten.
    """
    try:
        logger.info(f"Starting migration for org_id={org_id}")

        service = VespaMigrationService(org_id=org_id)
        stats = await service.migrate_all(dry_run=False)

        success = stats["projects"]["failed"] == 0 and stats["tasks"]["failed"] == 0

        return MigrationResponse(
            success=success,
            message=(
                f"Migration completed: "
                f"{stats['projects']['migrated']}/{stats['projects']['total']} projects, "
                f"{stats['tasks']['migrated']}/{stats['tasks']['total']} tasks"
            ),
            stats=stats,
        )

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validate")
async def validate_migration(
    org_id: str = Query("default", description="Organization ID"),
) -> ValidationResponse:
    """Validate migration by comparing Supabase and Vespa data counts.

    Returns counts from both backends to verify migration success.
    """
    try:
        service = VespaMigrationService(org_id=org_id)
        result = await service.validate_migration()

        return ValidationResponse(**result)

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
