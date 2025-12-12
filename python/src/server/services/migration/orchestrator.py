"""Migration orchestrator for data migrations to Vespa.

Coordinates the migration process including:
- Connection to source database
- Schema discovery and mapping
- Batch processing with embedding generation
- Progress tracking and checkpointing
"""

import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from .source_adapter import (
    ConnectionConfig,
    SourceAdapter,
    SourceSchema,
)
from .adapters.postgresql_adapter import PostgreSQLAdapter
from .adapters.supabase_adapter import SupabaseAdapter, SupabaseConnectionConfig
from .field_mapper import (
    FieldMapper,
    TableMapping,
    suggest_mappings,
)

logger = logging.getLogger(__name__)


class MigrationStatus(str, Enum):
    """Migration job status."""
    PENDING = "pending"
    CONNECTING = "connecting"
    DISCOVERING = "discovering"
    MAPPING = "mapping"
    VALIDATING = "validating"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TableProgress(BaseModel):
    """Progress tracking for a single table migration."""
    table_name: str
    total_rows: int = 0
    migrated_rows: int = 0
    failed_rows: int = 0
    current_cursor: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: MigrationStatus = MigrationStatus.PENDING
    errors: list[dict] = Field(default_factory=list)


class MigrationProgress(BaseModel):
    """Overall migration progress."""
    job_id: str
    status: MigrationStatus = MigrationStatus.PENDING
    tables: dict[str, TableProgress] = Field(default_factory=dict)
    total_rows: int = 0
    migrated_rows: int = 0
    failed_rows: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    errors: list[dict] = Field(default_factory=list)

    @property
    def progress_percent(self) -> float:
        """Get overall progress percentage."""
        if self.total_rows == 0:
            return 0.0
        return (self.migrated_rows / self.total_rows) * 100

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get migration duration in seconds."""
        if not self.started_at:
            return None
        end = self.completed_at or datetime.now(timezone.utc)
        return (end - self.started_at).total_seconds()


class MigrationConfig(BaseModel):
    """Configuration for a migration job."""
    job_id: str
    source_type: str  # postgresql, supabase
    connection_config: dict  # ConnectionConfig fields
    org_id: str = "default"
    workspace_id: str = "default"
    table_mappings: list[TableMapping] = Field(default_factory=list)
    batch_size: int = 1000
    max_concurrent_batches: int = 3
    generate_embeddings: bool = True
    dry_run: bool = False
    include_tables: Optional[list[str]] = None
    exclude_tables: Optional[list[str]] = None


class MigrationOrchestrator:
    """Orchestrates data migration from source database to Vespa.

    Handles connection management, schema discovery, field mapping,
    and batch processing with optional embedding generation.
    """

    def __init__(
        self,
        config: MigrationConfig,
        vespa_project_repo: Optional[Any] = None,
        vespa_task_repo: Optional[Any] = None,
        vespa_memory_repo: Optional[Any] = None,
        embedding_provider: Optional[Any] = None,
    ) -> None:
        """Initialize migration orchestrator.

        Args:
            config: Migration configuration
            vespa_project_repo: Vespa project repository (optional)
            vespa_task_repo: Vespa task repository (optional)
            vespa_memory_repo: Vespa memory repository (optional)
            embedding_provider: Embedding provider for vector generation
        """
        self.config = config
        self.project_repo = vespa_project_repo
        self.task_repo = vespa_task_repo
        self.memory_repo = vespa_memory_repo
        self.embedding_provider = embedding_provider

        self._adapter: Optional[SourceAdapter] = None
        self._schema: Optional[SourceSchema] = None
        self._field_mappers: dict[str, FieldMapper] = {}

        self.progress = MigrationProgress(job_id=config.job_id)

        # Callbacks for progress updates
        self._progress_callbacks: list[callable] = []

    @property
    def adapter(self) -> SourceAdapter:
        """Get the source adapter, creating if needed."""
        if not self._adapter:
            if self.config.source_type == "postgresql":
                self._adapter = PostgreSQLAdapter()
            elif self.config.source_type == "supabase":
                self._adapter = SupabaseAdapter()
            else:
                raise ValueError(f"Unknown source type: {self.config.source_type}")
        return self._adapter

    def on_progress(self, callback: callable) -> None:
        """Register a callback for progress updates.

        Args:
            callback: Function to call with MigrationProgress
        """
        self._progress_callbacks.append(callback)

    def _emit_progress(self) -> None:
        """Emit progress update to all callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(self.progress)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    async def connect(self) -> bool:
        """Connect to the source database.

        Returns:
            True if connection successful
        """
        self.progress.status = MigrationStatus.CONNECTING
        self._emit_progress()

        try:
            # Build connection config
            if self.config.source_type == "supabase":
                conn_config = SupabaseConnectionConfig(
                    host="",
                    port=443,
                    database="postgres",
                    username="",
                    password="",
                    **self.config.connection_config,
                )
            else:
                conn_config = ConnectionConfig(**self.config.connection_config)

            await self.adapter.connect(conn_config)

            logger.info(
                "migration_connected",
                extra={
                    "job_id": self.config.job_id,
                    "source_type": self.config.source_type,
                }
            )
            return True

        except Exception as e:
            self.progress.status = MigrationStatus.FAILED
            self.progress.errors.append({
                "phase": "connect",
                "error": str(e),
            })
            self._emit_progress()
            raise

    async def discover_schema(self) -> SourceSchema:
        """Discover schema from source database.

        Returns:
            Discovered schema
        """
        self.progress.status = MigrationStatus.DISCOVERING
        self._emit_progress()

        try:
            self._schema = await self.adapter.discover_schema(
                include_tables=self.config.include_tables,
                exclude_tables=self.config.exclude_tables,
            )

            # Initialize progress tracking for each table
            self.progress.total_rows = 0
            for table in self._schema.tables:
                self.progress.tables[table.name] = TableProgress(
                    table_name=table.name,
                    total_rows=table.estimated_rows,
                )
                self.progress.total_rows += table.estimated_rows

            self._emit_progress()

            logger.info(
                "schema_discovered",
                extra={
                    "job_id": self.config.job_id,
                    "tables": len(self._schema.tables),
                    "total_rows": self.progress.total_rows,
                }
            )

            return self._schema

        except Exception as e:
            self.progress.status = MigrationStatus.FAILED
            self.progress.errors.append({
                "phase": "discover_schema",
                "error": str(e),
            })
            self._emit_progress()
            raise

    async def generate_mappings(self) -> list[TableMapping]:
        """Generate suggested field mappings for discovered schema.

        Returns:
            List of suggested TableMapping configurations
        """
        self.progress.status = MigrationStatus.MAPPING
        self._emit_progress()

        if not self._schema:
            raise ValueError("Schema not discovered. Call discover_schema() first.")

        mappings = []

        for table in self._schema.tables:
            # Determine target schema based on table name
            target_schema = self._infer_target_schema(table.name)

            # Generate suggested mapping
            mapping = suggest_mappings(table.fields, target_schema)
            mapping.source_table = table.name

            # Find primary key mapping
            pk_field = next(
                (f for f in table.fields if f.is_primary_key),
                None
            )
            if pk_field:
                mapping.primary_key_mapping = pk_field.name

            mappings.append(mapping)

        return mappings

    def _infer_target_schema(self, table_name: str) -> str:
        """Infer Vespa target schema from table name."""
        # Map common table names to Oceanic schemas
        schema_map = {
            "projects": "oceanic_project",
            "archon_projects": "oceanic_project",
            "tasks": "oceanic_task",
            "archon_tasks": "oceanic_task",
            "memories": "oceanic_memory",
            "agent_memory": "oceanic_memory",
            "documents": "oceanic_document",
            "work_orders": "oceanic_work_order",
        }

        if table_name in schema_map:
            return schema_map[table_name]

        # Default: prefix with oceanic_
        return f"oceanic_{table_name}"

    def set_mappings(self, mappings: list[TableMapping]) -> None:
        """Set the field mappings to use for migration.

        Args:
            mappings: List of TableMapping configurations
        """
        self.config.table_mappings = mappings

        # Create field mappers
        for mapping in mappings:
            self._field_mappers[mapping.source_table] = FieldMapper(mapping)

    async def validate(self) -> dict:
        """Validate migration configuration.

        Returns:
            Validation results with any warnings or errors
        """
        self.progress.status = MigrationStatus.VALIDATING
        self._emit_progress()

        results = {
            "valid": True,
            "warnings": [],
            "errors": [],
        }

        # Check mappings exist
        if not self.config.table_mappings:
            results["errors"].append("No table mappings configured")
            results["valid"] = False

        # Check each mapping
        for mapping in self.config.table_mappings:
            # Verify source table exists in schema
            if self._schema:
                table = next(
                    (t for t in self._schema.tables if t.name == mapping.source_table),
                    None
                )
                if not table:
                    results["errors"].append(
                        f"Source table not found: {mapping.source_table}"
                    )
                    results["valid"] = False
                    continue

                # Check all source fields exist
                source_fields = {f.name for f in table.fields}
                for field_map in mapping.field_mappings:
                    if field_map.source_field not in source_fields:
                        results["warnings"].append(
                            f"Source field not found: {mapping.source_table}.{field_map.source_field}"
                        )

        # Check embedding provider if embeddings are enabled
        if self.config.generate_embeddings and not self.embedding_provider:
            results["warnings"].append(
                "Embedding generation enabled but no provider configured"
            )

        return results

    async def run(self) -> MigrationProgress:
        """Run the full migration.

        Returns:
            Final migration progress
        """
        self.progress.status = MigrationStatus.RUNNING
        self.progress.started_at = datetime.now(timezone.utc)
        self._emit_progress()

        try:
            for mapping in self.config.table_mappings:
                await self._migrate_table(mapping)

                if self.progress.status == MigrationStatus.CANCELLED:
                    break

            if self.progress.status != MigrationStatus.CANCELLED:
                self.progress.status = MigrationStatus.COMPLETED

            self.progress.completed_at = datetime.now(timezone.utc)
            self._emit_progress()

            logger.info(
                "migration_completed",
                extra={
                    "job_id": self.config.job_id,
                    "migrated_rows": self.progress.migrated_rows,
                    "failed_rows": self.progress.failed_rows,
                    "duration_seconds": self.progress.duration_seconds,
                }
            )

            return self.progress

        except Exception as e:
            self.progress.status = MigrationStatus.FAILED
            self.progress.completed_at = datetime.now(timezone.utc)
            self.progress.errors.append({
                "phase": "run",
                "error": str(e),
            })
            self._emit_progress()
            raise

    async def _migrate_table(self, mapping: TableMapping) -> None:
        """Migrate a single table.

        Args:
            mapping: Table mapping configuration
        """
        table_name = mapping.source_table
        table_progress = self.progress.tables.get(table_name)

        if not table_progress:
            table_progress = TableProgress(table_name=table_name)
            self.progress.tables[table_name] = table_progress

        table_progress.status = MigrationStatus.RUNNING
        table_progress.started_at = datetime.now(timezone.utc)
        self._emit_progress()

        field_mapper = self._field_mappers.get(table_name)
        if not field_mapper:
            field_mapper = FieldMapper(mapping)

        try:
            # Check if table has vector column for special handling
            has_vector = any(
                fm.target_field == "embedding" for fm in mapping.field_mappings
            )

            # Stream data from source
            cursor = table_progress.current_cursor

            if has_vector:
                # Find vector column name
                vector_col = next(
                    (fm.source_field for fm in mapping.field_mappings
                     if fm.target_field == "embedding"),
                    "embedding"
                )
                data_iterator = self.adapter.export_with_vectors(
                    table_name,
                    vector_column=vector_col,
                    batch_size=self.config.batch_size,
                    cursor=cursor,
                )
            else:
                data_iterator = self.adapter.export_table(
                    table_name,
                    batch_size=self.config.batch_size,
                    cursor=cursor,
                )

            async for batch, next_cursor in data_iterator:
                if self.progress.status == MigrationStatus.CANCELLED:
                    break

                await self._process_batch(
                    batch,
                    mapping,
                    field_mapper,
                    table_progress,
                )

                table_progress.current_cursor = next_cursor
                self._emit_progress()

            if self.progress.status != MigrationStatus.CANCELLED:
                table_progress.status = MigrationStatus.COMPLETED
                table_progress.completed_at = datetime.now(timezone.utc)
                self._emit_progress()

        except Exception as e:
            table_progress.status = MigrationStatus.FAILED
            table_progress.errors.append({
                "error": str(e),
            })
            self.progress.errors.append({
                "table": table_name,
                "error": str(e),
            })
            self._emit_progress()
            raise

    async def _process_batch(
        self,
        batch: list[dict],
        mapping: TableMapping,
        field_mapper: FieldMapper,
        table_progress: TableProgress,
    ) -> None:
        """Process a batch of records.

        Args:
            batch: List of source records
            mapping: Table mapping configuration
            field_mapper: Field mapper instance
            table_progress: Progress tracker for this table
        """
        if self.config.dry_run:
            # Just count records in dry run
            table_progress.migrated_rows += len(batch)
            self.progress.migrated_rows += len(batch)
            return

        for record in batch:
            try:
                # Transform record
                transformed = field_mapper.transform_row(record)

                # Add multi-tenancy fields
                transformed["org_id"] = self.config.org_id
                transformed["workspace_id"] = self.config.workspace_id

                # Generate embedding if configured
                if self.config.generate_embeddings and self.embedding_provider:
                    embedding_text = field_mapper.get_embedding_text(record)
                    if embedding_text:
                        embedding = await self.embedding_provider.embed(embedding_text)
                        transformed["embedding"] = embedding

                # Write to appropriate Vespa repo based on target schema
                await self._write_to_vespa(mapping.target_schema, transformed)

                table_progress.migrated_rows += 1
                self.progress.migrated_rows += 1

            except Exception as e:
                table_progress.failed_rows += 1
                self.progress.failed_rows += 1
                table_progress.errors.append({
                    "record_id": record.get("id", "unknown"),
                    "error": str(e),
                })

                logger.warning(
                    "record_migration_failed",
                    extra={
                        "table": mapping.source_table,
                        "record_id": record.get("id"),
                        "error": str(e),
                    }
                )

    async def _write_to_vespa(self, target_schema: str, data: dict) -> None:
        """Write transformed data to appropriate Vespa repository.

        Args:
            target_schema: Target Vespa schema name
            data: Transformed data to write
        """
        if target_schema == "oceanic_project" and self.project_repo:
            await self.project_repo.create(data)
        elif target_schema == "oceanic_task" and self.task_repo:
            await self.task_repo.create(data)
        elif target_schema == "oceanic_memory" and self.memory_repo:
            await self.memory_repo.create(data)
        else:
            # Generic write via Vespa client
            logger.warning(
                f"No repository configured for schema: {target_schema}"
            )

    async def pause(self) -> None:
        """Pause the migration."""
        if self.progress.status == MigrationStatus.RUNNING:
            self.progress.status = MigrationStatus.PAUSED
            self._emit_progress()

    async def resume(self) -> MigrationProgress:
        """Resume a paused migration.

        Returns:
            Updated migration progress
        """
        if self.progress.status == MigrationStatus.PAUSED:
            return await self.run()
        return self.progress

    async def cancel(self) -> None:
        """Cancel the migration."""
        self.progress.status = MigrationStatus.CANCELLED
        self.progress.completed_at = datetime.now(timezone.utc)
        self._emit_progress()

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._adapter:
            await self._adapter.disconnect()
            self._adapter = None
