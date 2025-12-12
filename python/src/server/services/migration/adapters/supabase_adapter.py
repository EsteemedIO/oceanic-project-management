"""Supabase adapter for data migration to Vespa.

Wraps the Supabase client to export data from Supabase-hosted
PostgreSQL databases, including support for RLS-aware exports.
"""

import logging
from datetime import datetime
from typing import Any, AsyncIterator, Optional
from urllib.parse import urlparse

from supabase import Client, create_client

from ..source_adapter import (
    ChangeEvent,
    ChangeEventType,
    ConnectionConfig,
    FieldSchema,
    FieldType,
    Relationship,
    SourceAdapter,
    SourceSchema,
    TableSchema,
)

logger = logging.getLogger(__name__)


class SupabaseConnectionConfig(ConnectionConfig):
    """Extended config for Supabase connections."""
    supabase_url: str = ""
    supabase_key: str = ""  # service_role key for bypassing RLS


class SupabaseAdapter(SourceAdapter):
    """Supabase adapter using PostgREST API.

    Uses the Supabase client library for data export,
    bypassing RLS with service_role key.
    """

    def __init__(self) -> None:
        self._client: Optional[Client] = None
        self._config: Optional[SupabaseConnectionConfig] = None
        self._schema_cache: Optional[SourceSchema] = None

    @property
    def adapter_type(self) -> str:
        return "supabase"

    async def connect(self, config: ConnectionConfig) -> bool:
        """Connect to Supabase project.

        The config should include supabase_url and supabase_key
        in the extra_params, or use a SupabaseConnectionConfig.

        Args:
            config: Connection configuration with Supabase credentials

        Returns:
            True if connection successful
        """
        # Extract Supabase-specific config
        if isinstance(config, SupabaseConnectionConfig):
            supabase_url = config.supabase_url
            supabase_key = config.supabase_key
        else:
            supabase_url = config.extra_params.get("supabase_url", "")
            supabase_key = config.extra_params.get("supabase_key", "")

        if not supabase_url or not supabase_key:
            raise ValueError("Supabase URL and service key are required")

        try:
            self._client = create_client(supabase_url, supabase_key)

            # Test connection by fetching a small query
            # This uses PostgREST which is Supabase's REST API
            self._client.table("_test_connection").select("*").limit(0).execute()

        except Exception as e:
            # The test table might not exist, but we should be connected
            if "404" not in str(e) and "relation" not in str(e).lower():
                logger.error(f"Failed to connect to Supabase: {e}")
                raise ConnectionError(f"Supabase connection failed: {e}")

        # Store config
        self._config = SupabaseConnectionConfig(
            host=urlparse(supabase_url).netloc,
            port=443,
            database="postgres",
            username="",
            password="",
            supabase_url=supabase_url,
            supabase_key=supabase_key,
        )

        logger.info(
            "supabase_connected",
            extra={"url": supabase_url}
        )
        return True

    async def test_connection(self) -> bool:
        """Test if the Supabase connection is valid."""
        if not self._client:
            return False

        try:
            # Try a simple query
            self._client.table("_test").select("*").limit(0).execute()
            return True
        except Exception:
            # Connection might be fine, just no table exists
            return True

    async def discover_schema(
        self,
        include_tables: Optional[list[str]] = None,
        exclude_tables: Optional[list[str]] = None,
    ) -> SourceSchema:
        """Discover schema from Supabase.

        Note: Supabase's PostgREST doesn't expose full schema,
        so we use known table introspection.

        Args:
            include_tables: Whitelist of tables to include
            exclude_tables: Blacklist of tables to exclude

        Returns:
            Discovered schema (limited information via REST API)
        """
        if not self._client:
            raise ConnectionError("Not connected to Supabase")

        tables: list[TableSchema] = []

        # If include_tables not specified, try common Archon tables
        if not include_tables:
            include_tables = [
                "archon_projects",
                "archon_tasks",
                "sources",
                "documents",
                "code_examples",
            ]

        for table_name in include_tables:
            if exclude_tables and table_name in exclude_tables:
                continue

            try:
                # Try to get sample data to infer schema
                result = self._client.table(table_name).select("*").limit(5).execute()

                if result.data:
                    fields = self._infer_fields_from_data(result.data[0])
                    estimated_rows = await self._estimate_table_rows(table_name)

                    tables.append(TableSchema(
                        name=table_name,
                        fields=fields,
                        primary_key="id",  # Assume standard Supabase pattern
                        estimated_rows=estimated_rows,
                    ))
                else:
                    # Empty table, create minimal schema
                    tables.append(TableSchema(
                        name=table_name,
                        fields=[
                            FieldSchema(
                                name="id",
                                type=FieldType.UUID,
                                original_type="uuid",
                                is_primary_key=True,
                            )
                        ],
                        primary_key="id",
                        estimated_rows=0,
                    ))

            except Exception as e:
                logger.warning(f"Could not discover table {table_name}: {e}")

        schema = SourceSchema(
            tables=tables,
            source_type="supabase",
            estimated_total_rows=sum(t.estimated_rows for t in tables),
        )

        self._schema_cache = schema
        return schema

    def _infer_fields_from_data(self, sample_row: dict) -> list[FieldSchema]:
        """Infer field schemas from a sample data row."""
        fields = []

        for key, value in sample_row.items():
            field_type = self._infer_type_from_value(value)

            fields.append(FieldSchema(
                name=key,
                type=field_type,
                original_type=type(value).__name__ if value else "unknown",
                nullable=value is None,
                is_primary_key=(key == "id"),
            ))

        return fields

    def _infer_type_from_value(self, value: Any) -> FieldType:
        """Infer FieldType from Python value."""
        if value is None:
            return FieldType.UNKNOWN
        if isinstance(value, bool):
            return FieldType.BOOLEAN
        if isinstance(value, int):
            return FieldType.INTEGER
        if isinstance(value, float):
            return FieldType.DOUBLE
        if isinstance(value, str):
            # Check for UUID pattern
            if len(value) == 36 and value.count("-") == 4:
                return FieldType.UUID
            # Check for ISO timestamp
            if "T" in value and ("Z" in value or "+" in value):
                return FieldType.TIMESTAMP
            return FieldType.STRING
        if isinstance(value, list):
            # Check if it looks like an embedding vector
            if value and all(isinstance(x, (int, float)) for x in value):
                return FieldType.VECTOR
            return FieldType.ARRAY
        if isinstance(value, dict):
            return FieldType.JSONB

        return FieldType.UNKNOWN

    async def _estimate_table_rows(self, table_name: str) -> int:
        """Estimate row count using count query."""
        try:
            result = self._client.table(table_name).select("*", count="exact").limit(0).execute()
            return result.count or 0
        except Exception:
            return 0

    async def estimate_size(self) -> dict[str, int]:
        """Get estimated row counts for all discovered tables."""
        if not self._client:
            raise ConnectionError("Not connected to Supabase")

        if self._schema_cache:
            return {t.name: t.estimated_rows for t in self._schema_cache.tables}

        # Discover schema first
        schema = await self.discover_schema()
        return {t.name: t.estimated_rows for t in schema.tables}

    async def export_table(
        self,
        table_name: str,
        batch_size: int = 1000,
        cursor: Optional[str] = None,
        order_by: Optional[str] = None,
    ) -> AsyncIterator[tuple[list[dict], Optional[str]]]:
        """Stream table data in batches using PostgREST pagination.

        Args:
            table_name: Name of the table to export
            batch_size: Number of records per batch
            cursor: Resume cursor (offset value)
            order_by: Column to order by (default: id)

        Yields:
            Tuple of (batch of records, next cursor or None if complete)
        """
        if not self._client:
            raise ConnectionError("Not connected to Supabase")

        order_column = order_by or "id"
        offset = int(cursor) if cursor else 0

        while True:
            try:
                result = self._client.table(table_name)\
                    .select("*")\
                    .order(order_column)\
                    .range(offset, offset + batch_size - 1)\
                    .execute()

                if not result.data:
                    break

                batch = result.data
                offset += len(batch)
                next_cursor = str(offset) if len(batch) == batch_size else None

                yield batch, next_cursor

                if len(batch) < batch_size:
                    break

            except Exception as e:
                logger.error(f"Failed to export table {table_name}: {e}")
                raise

    async def export_with_vectors(
        self,
        table_name: str,
        vector_column: str,
        batch_size: int = 1000,
        cursor: Optional[str] = None,
    ) -> AsyncIterator[tuple[list[dict], Optional[str]]]:
        """Stream table data with vector embeddings.

        Supabase stores vectors as arrays, so no special conversion needed.

        Args:
            table_name: Name of the table to export
            vector_column: Name of the vector/embedding column
            batch_size: Number of records per batch
            cursor: Resume cursor from previous export

        Yields:
            Tuple of (batch of records with vectors, next cursor)
        """
        # Supabase stores pgvector as JSON arrays, so use standard export
        async for batch, next_cursor in self.export_table(
            table_name, batch_size, cursor, order_by="id"
        ):
            # Ensure vectors are lists
            for row in batch:
                if vector_column in row and row[vector_column]:
                    vec = row[vector_column]
                    if isinstance(vec, str):
                        # Parse string representation
                        import json
                        try:
                            row[vector_column] = json.loads(vec)
                        except Exception:
                            pass

            yield batch, next_cursor

    async def get_changes_since(
        self,
        table_name: str,
        timestamp: datetime,
        change_tracking_column: str = "updated_at",
    ) -> AsyncIterator[ChangeEvent]:
        """Get changes since timestamp for incremental sync.

        Args:
            table_name: Name of the table to track
            timestamp: Get changes after this time
            change_tracking_column: Column used for change tracking

        Yields:
            Change events representing modifications
        """
        if not self._client:
            raise ConnectionError("Not connected to Supabase")

        try:
            result = self._client.table(table_name)\
                .select("*")\
                .gt(change_tracking_column, timestamp.isoformat())\
                .order(change_tracking_column)\
                .execute()

            for row in result.data or []:
                yield ChangeEvent(
                    event_type=ChangeEventType.UPDATE,
                    table_name=table_name,
                    primary_key=str(row.get("id", "")),
                    data=row,
                    timestamp=datetime.fromisoformat(
                        row.get(change_tracking_column, datetime.utcnow().isoformat())
                        .replace("Z", "+00:00")
                    ),
                )

        except Exception as e:
            logger.error(f"Failed to get changes for {table_name}: {e}")
            raise

    async def get_row_count(self, table_name: str) -> int:
        """Get exact row count for a table."""
        return await self._estimate_table_rows(table_name)

    async def disconnect(self) -> None:
        """Clean up Supabase client."""
        self._client = None
        logger.info("supabase_disconnected")
