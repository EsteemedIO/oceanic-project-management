"""PostgreSQL adapter for data migration to Vespa.

Supports direct PostgreSQL connections using psycopg3 (async),
including pgvector extension for vector embeddings.
"""

import json
import logging
from datetime import datetime
from typing import Any, AsyncIterator, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

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

# Mapping PostgreSQL types to standardized FieldType
PG_TYPE_MAP: dict[str, FieldType] = {
    # String types
    "character varying": FieldType.STRING,
    "varchar": FieldType.STRING,
    "character": FieldType.STRING,
    "char": FieldType.STRING,
    "text": FieldType.TEXT,
    "name": FieldType.STRING,
    # Numeric types
    "smallint": FieldType.INTEGER,
    "integer": FieldType.INTEGER,
    "int": FieldType.INTEGER,
    "int4": FieldType.INTEGER,
    "bigint": FieldType.BIGINT,
    "int8": FieldType.BIGINT,
    "real": FieldType.FLOAT,
    "float4": FieldType.FLOAT,
    "double precision": FieldType.DOUBLE,
    "float8": FieldType.DOUBLE,
    "numeric": FieldType.DOUBLE,
    "decimal": FieldType.DOUBLE,
    # Boolean
    "boolean": FieldType.BOOLEAN,
    "bool": FieldType.BOOLEAN,
    # Date/Time
    "timestamp without time zone": FieldType.TIMESTAMP,
    "timestamp with time zone": FieldType.TIMESTAMP,
    "timestamp": FieldType.TIMESTAMP,
    "timestamptz": FieldType.TIMESTAMP,
    "date": FieldType.DATE,
    "time": FieldType.STRING,
    "time without time zone": FieldType.STRING,
    "time with time zone": FieldType.STRING,
    "interval": FieldType.STRING,
    # JSON types
    "json": FieldType.JSON,
    "jsonb": FieldType.JSONB,
    # Array types (handled specially)
    "ARRAY": FieldType.ARRAY,
    # UUID
    "uuid": FieldType.UUID,
    # Binary
    "bytea": FieldType.BINARY,
    # Vector (pgvector)
    "vector": FieldType.VECTOR,
}


class PostgreSQLAdapter(SourceAdapter):
    """PostgreSQL adapter with pgvector support.

    Uses psycopg3 async driver for efficient streaming
    and pgvector for vector column handling.
    """

    def __init__(self) -> None:
        self._pool: Optional[AsyncConnectionPool] = None
        self._config: Optional[ConnectionConfig] = None
        self._schema_cache: Optional[SourceSchema] = None

    @property
    def adapter_type(self) -> str:
        return "postgresql"

    async def connect(self, config: ConnectionConfig) -> bool:
        """Establish connection pool to PostgreSQL.

        Args:
            config: Connection configuration

        Returns:
            True if connection successful
        """
        self._config = config

        # Build connection string
        conninfo = (
            f"host={config.host} "
            f"port={config.port} "
            f"dbname={config.database} "
            f"user={config.username} "
            f"password={config.password} "
            f"sslmode={config.ssl_mode}"
        )

        try:
            # Create async connection pool
            self._pool = AsyncConnectionPool(
                conninfo=conninfo,
                min_size=1,
                max_size=10,
                kwargs={"row_factory": dict_row},
            )
            await self._pool.open()

            # Test connection
            async with self._pool.connection() as conn:
                await conn.execute("SELECT 1")

            logger.info(
                "postgresql_connected",
                extra={
                    "host": config.host,
                    "database": config.database,
                }
            )
            return True

        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise ConnectionError(f"PostgreSQL connection failed: {e}")

    async def test_connection(self) -> bool:
        """Test if the connection pool is healthy."""
        if not self._pool:
            return False

        try:
            async with self._pool.connection() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    async def discover_schema(
        self,
        include_tables: Optional[list[str]] = None,
        exclude_tables: Optional[list[str]] = None,
    ) -> SourceSchema:
        """Discover schema from PostgreSQL using pg_catalog.

        Args:
            include_tables: Whitelist of tables to include
            exclude_tables: Blacklist of tables to exclude

        Returns:
            Complete schema including tables, fields, and relationships
        """
        if not self._pool:
            raise ConnectionError("Not connected to database")

        tables: list[TableSchema] = []
        relationships: list[Relationship] = []
        total_rows = 0

        async with self._pool.connection() as conn:
            # Get all tables in public schema
            tables_result = await conn.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """)
            table_names = [row["table_name"] for row in await tables_result.fetchall()]

            # Filter tables
            if include_tables:
                table_names = [t for t in table_names if t in include_tables]
            if exclude_tables:
                table_names = [t for t in table_names if t not in exclude_tables]

            for table_name in table_names:
                table_schema = await self._discover_table_schema(conn, table_name)
                tables.append(table_schema)
                total_rows += table_schema.estimated_rows

            # Discover foreign key relationships
            relationships = await self._discover_relationships(conn, table_names)

        # Check for pgvector extension
        source_version = await self._get_pg_version()

        schema = SourceSchema(
            tables=tables,
            relationships=relationships,
            estimated_total_rows=total_rows,
            source_type="postgresql",
            source_version=source_version,
        )

        self._schema_cache = schema
        logger.info(
            "schema_discovered",
            extra={
                "tables": len(tables),
                "total_rows": total_rows,
            }
        )

        return schema

    async def _discover_table_schema(
        self,
        conn: psycopg.AsyncConnection,
        table_name: str,
    ) -> TableSchema:
        """Discover schema for a single table."""

        # Get columns
        columns_result = await conn.execute("""
            SELECT
                c.column_name,
                c.data_type,
                c.udt_name,
                c.is_nullable,
                c.character_maximum_length,
                CASE WHEN pk.column_name IS NOT NULL THEN true ELSE false END as is_primary
            FROM information_schema.columns c
            LEFT JOIN (
                SELECT ku.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage ku
                    ON tc.constraint_name = ku.constraint_name
                WHERE tc.table_name = %s
                  AND tc.constraint_type = 'PRIMARY KEY'
            ) pk ON c.column_name = pk.column_name
            WHERE c.table_name = %s
              AND c.table_schema = 'public'
            ORDER BY c.ordinal_position
        """, (table_name, table_name))

        fields: list[FieldSchema] = []
        primary_key: Optional[str] = None

        for row in await columns_result.fetchall():
            col_name = row["column_name"]
            data_type = row["data_type"]
            udt_name = row["udt_name"]
            is_nullable = row["is_nullable"] == "YES"
            is_primary = row["is_primary"]

            # Determine field type
            field_type = self._map_pg_type(data_type, udt_name)

            # Get vector dimension if applicable
            vector_dim = None
            if field_type == FieldType.VECTOR:
                vector_dim = await self._get_vector_dimension(conn, table_name, col_name)

            field = FieldSchema(
                name=col_name,
                type=field_type,
                original_type=udt_name or data_type,
                nullable=is_nullable,
                is_primary_key=is_primary,
                vector_dimension=vector_dim,
            )
            fields.append(field)

            if is_primary:
                primary_key = col_name

        # Get indexes
        indexes = await self._get_table_indexes(conn, table_name)

        # Estimate row count
        estimated_rows = await self._estimate_table_rows(conn, table_name)

        return TableSchema(
            name=table_name,
            fields=fields,
            primary_key=primary_key,
            indexes=indexes,
            estimated_rows=estimated_rows,
        )

    def _map_pg_type(self, data_type: str, udt_name: str) -> FieldType:
        """Map PostgreSQL type to standardized FieldType."""
        # Check UDT name first (for extensions like pgvector)
        if udt_name == "vector":
            return FieldType.VECTOR

        # Handle arrays
        if data_type == "ARRAY":
            return FieldType.ARRAY

        # Check standard mappings
        lookup_type = data_type.lower()
        if lookup_type in PG_TYPE_MAP:
            return PG_TYPE_MAP[lookup_type]

        logger.warning(f"Unknown PostgreSQL type: {data_type} ({udt_name})")
        return FieldType.UNKNOWN

    async def _get_vector_dimension(
        self,
        conn: psycopg.AsyncConnection,
        table_name: str,
        column_name: str,
    ) -> Optional[int]:
        """Get dimension of a vector column."""
        try:
            # Try to get from column definition
            result = await conn.execute(f"""
                SELECT atttypmod
                FROM pg_attribute a
                JOIN pg_class c ON a.attrelid = c.oid
                WHERE c.relname = %s AND a.attname = %s
            """, (table_name, column_name))

            row = await result.fetchone()
            if row and row["atttypmod"] > 0:
                return row["atttypmod"]

            # Fallback: sample a row
            result = await conn.execute(f"""
                SELECT vector_dims("{column_name}") as dim
                FROM "{table_name}"
                WHERE "{column_name}" IS NOT NULL
                LIMIT 1
            """)
            row = await result.fetchone()
            if row and row["dim"]:
                return row["dim"]

        except Exception as e:
            logger.warning(f"Could not determine vector dimension: {e}")

        return None

    async def _get_table_indexes(
        self,
        conn: psycopg.AsyncConnection,
        table_name: str,
    ) -> list[str]:
        """Get list of indexes for a table."""
        result = await conn.execute("""
            SELECT indexname
            FROM pg_indexes
            WHERE tablename = %s
        """, (table_name,))

        return [row["indexname"] for row in await result.fetchall()]

    async def _estimate_table_rows(
        self,
        conn: psycopg.AsyncConnection,
        table_name: str,
    ) -> int:
        """Estimate row count using pg_class statistics."""
        result = await conn.execute("""
            SELECT reltuples::bigint as estimate
            FROM pg_class
            WHERE relname = %s
        """, (table_name,))

        row = await result.fetchone()
        if row and row["estimate"] >= 0:
            return int(row["estimate"])

        # Fallback to COUNT for small tables
        result = await conn.execute(f'SELECT COUNT(*) as cnt FROM "{table_name}"')
        row = await result.fetchone()
        return row["cnt"] if row else 0

    async def _discover_relationships(
        self,
        conn: psycopg.AsyncConnection,
        table_names: list[str],
    ) -> list[Relationship]:
        """Discover foreign key relationships between tables."""
        result = await conn.execute("""
            SELECT
                tc.table_name as from_table,
                kcu.column_name as from_column,
                ccu.table_name as to_table,
                ccu.column_name as to_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage ccu
                ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_schema = 'public'
        """)

        relationships = []
        for row in await result.fetchall():
            if row["from_table"] in table_names and row["to_table"] in table_names:
                relationships.append(Relationship(
                    from_table=row["from_table"],
                    from_column=row["from_column"],
                    to_table=row["to_table"],
                    to_column=row["to_column"],
                ))

        return relationships

    async def _get_pg_version(self) -> str:
        """Get PostgreSQL version string."""
        if not self._pool:
            return "unknown"

        async with self._pool.connection() as conn:
            result = await conn.execute("SELECT version()")
            row = await result.fetchone()
            return row["version"] if row else "unknown"

    async def estimate_size(self) -> dict[str, int]:
        """Get estimated row counts for all tables."""
        if not self._pool:
            raise ConnectionError("Not connected to database")

        if self._schema_cache:
            return {t.name: t.estimated_rows for t in self._schema_cache.tables}

        async with self._pool.connection() as conn:
            result = await conn.execute("""
                SELECT relname as table_name, reltuples::bigint as estimate
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'public'
                  AND c.relkind = 'r'
            """)

            return {
                row["table_name"]: max(0, int(row["estimate"]))
                for row in await result.fetchall()
            }

    async def export_table(
        self,
        table_name: str,
        batch_size: int = 1000,
        cursor: Optional[str] = None,
        order_by: Optional[str] = None,
    ) -> AsyncIterator[tuple[list[dict], Optional[str]]]:
        """Stream table data in batches using cursor-based pagination.

        Args:
            table_name: Name of the table to export
            batch_size: Number of records per batch
            cursor: Resume cursor (last primary key value)
            order_by: Column to order by (default: primary key)

        Yields:
            Tuple of (batch of records, next cursor or None if complete)
        """
        if not self._pool:
            raise ConnectionError("Not connected to database")

        # Determine order column
        if not order_by:
            order_by = await self._get_primary_key(table_name)

        if not order_by:
            raise ValueError(f"No primary key or order_by column for table {table_name}")

        async with self._pool.connection() as conn:
            offset = 0
            if cursor:
                # Cursor is the last seen value for cursor-based pagination
                # For simplicity, we use offset-based here
                try:
                    offset = int(cursor)
                except ValueError:
                    offset = 0

            while True:
                # Build query with pagination
                query = f"""
                    SELECT *
                    FROM "{table_name}"
                    ORDER BY "{order_by}"
                    LIMIT %s OFFSET %s
                """

                result = await conn.execute(query, (batch_size, offset))
                rows = await result.fetchall()

                if not rows:
                    break

                # Convert rows to dicts, handling special types
                batch = [self._convert_row(dict(row)) for row in rows]

                offset += len(rows)
                next_cursor = str(offset) if len(rows) == batch_size else None

                yield batch, next_cursor

                if len(rows) < batch_size:
                    break

    async def export_with_vectors(
        self,
        table_name: str,
        vector_column: str,
        batch_size: int = 1000,
        cursor: Optional[str] = None,
    ) -> AsyncIterator[tuple[list[dict], Optional[str]]]:
        """Stream table data with vector embeddings.

        Specifically handles pgvector columns by converting them
        to Python lists suitable for Vespa ingestion.

        Args:
            table_name: Name of the table to export
            vector_column: Name of the vector/embedding column
            batch_size: Number of records per batch
            cursor: Resume cursor from previous export

        Yields:
            Tuple of (batch of records with vectors as lists, next cursor)
        """
        if not self._pool:
            raise ConnectionError("Not connected to database")

        pk_column = await self._get_primary_key(table_name)
        if not pk_column:
            raise ValueError(f"No primary key for table {table_name}")

        async with self._pool.connection() as conn:
            # Register pgvector types for proper conversion
            try:
                await conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            except Exception:
                logger.warning("pgvector extension may not be installed")

            offset = int(cursor) if cursor else 0

            while True:
                # Select all columns, converting vector to array format
                query = f"""
                    SELECT *,
                           "{vector_column}"::text as _vector_text
                    FROM "{table_name}"
                    ORDER BY "{pk_column}"
                    LIMIT %s OFFSET %s
                """

                result = await conn.execute(query, (batch_size, offset))
                rows = await result.fetchall()

                if not rows:
                    break

                batch = []
                for row in rows:
                    row_dict = dict(row)

                    # Parse vector from text representation [1.0, 2.0, ...]
                    vector_text = row_dict.pop("_vector_text", None)
                    if vector_text:
                        try:
                            # pgvector format: [1.0,2.0,3.0]
                            vector_str = vector_text.strip("[]")
                            row_dict[vector_column] = [
                                float(x) for x in vector_str.split(",")
                            ]
                        except Exception as e:
                            logger.warning(f"Failed to parse vector: {e}")
                            row_dict[vector_column] = None

                    batch.append(self._convert_row(row_dict))

                offset += len(rows)
                next_cursor = str(offset) if len(rows) == batch_size else None

                yield batch, next_cursor

                if len(rows) < batch_size:
                    break

    def _convert_row(self, row: dict) -> dict:
        """Convert PostgreSQL row values to JSON-serializable format."""
        converted = {}
        for key, value in row.items():
            if value is None:
                converted[key] = None
            elif isinstance(value, datetime):
                converted[key] = value.isoformat()
            elif isinstance(value, (dict, list)):
                converted[key] = value  # Already JSON-compatible
            elif isinstance(value, bytes):
                # Convert bytea to base64
                import base64
                converted[key] = base64.b64encode(value).decode("utf-8")
            else:
                converted[key] = value

        return converted

    async def get_changes_since(
        self,
        table_name: str,
        timestamp: datetime,
        change_tracking_column: str = "updated_at",
    ) -> AsyncIterator[ChangeEvent]:
        """Get changes since timestamp for incremental sync.

        Requires the table to have a timestamp column (e.g., updated_at)
        that is updated on each modification.

        Args:
            table_name: Name of the table to track
            timestamp: Get changes after this time
            change_tracking_column: Column used for change tracking

        Yields:
            Change events representing modifications
        """
        if not self._pool:
            raise ConnectionError("Not connected to database")

        pk_column = await self._get_primary_key(table_name)
        if not pk_column:
            raise ValueError(f"No primary key for table {table_name}")

        async with self._pool.connection() as conn:
            # Get updated/inserted rows
            query = f"""
                SELECT *
                FROM "{table_name}"
                WHERE "{change_tracking_column}" > %s
                ORDER BY "{change_tracking_column}"
            """

            result = await conn.execute(query, (timestamp,))

            async for row in result:
                row_dict = self._convert_row(dict(row))

                yield ChangeEvent(
                    event_type=ChangeEventType.UPDATE,  # Can't distinguish insert vs update
                    table_name=table_name,
                    primary_key=str(row_dict.get(pk_column)),
                    data=row_dict,
                    timestamp=row_dict.get(change_tracking_column, datetime.utcnow()),
                )

    async def get_row_count(self, table_name: str) -> int:
        """Get exact row count for a table."""
        if not self._pool:
            raise ConnectionError("Not connected to database")

        async with self._pool.connection() as conn:
            result = await conn.execute(f'SELECT COUNT(*) as cnt FROM "{table_name}"')
            row = await result.fetchone()
            return row["cnt"] if row else 0

    async def _get_primary_key(self, table_name: str) -> Optional[str]:
        """Get primary key column name for a table."""
        if not self._pool:
            return None

        async with self._pool.connection() as conn:
            result = await conn.execute("""
                SELECT a.attname
                FROM pg_index i
                JOIN pg_attribute a ON a.attrelid = i.indrelid
                    AND a.attnum = ANY(i.indkey)
                WHERE i.indrelid = %s::regclass
                  AND i.indisprimary
            """, (table_name,))

            row = await result.fetchone()
            return row["attname"] if row else None

    async def disconnect(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("postgresql_disconnected")
