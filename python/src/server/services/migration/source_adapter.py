"""Abstract base class for source database adapters.

Provides the interface that all source adapters must implement
for migrating data to Vespa.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Optional

from pydantic import BaseModel, Field


class FieldType(str, Enum):
    """Standard field types for cross-platform mapping."""
    STRING = "string"
    TEXT = "text"
    INTEGER = "integer"
    BIGINT = "bigint"
    FLOAT = "float"
    DOUBLE = "double"
    BOOLEAN = "boolean"
    TIMESTAMP = "timestamp"
    DATE = "date"
    JSON = "json"
    JSONB = "jsonb"
    ARRAY = "array"
    VECTOR = "vector"
    UUID = "uuid"
    BINARY = "binary"
    UNKNOWN = "unknown"


class FieldSchema(BaseModel):
    """Schema for a single field/column."""
    name: str
    type: FieldType
    original_type: str  # Original database type
    nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_key_ref: Optional[str] = None  # table.column format
    sample_values: list[Any] = Field(default_factory=list)
    vector_dimension: Optional[int] = None  # For vector fields


class TableSchema(BaseModel):
    """Schema for a database table/collection."""
    name: str
    fields: list[FieldSchema]
    primary_key: Optional[str] = None
    indexes: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    estimated_rows: int = 0


class Relationship(BaseModel):
    """Represents a relationship between tables."""
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    relationship_type: str = "many_to_one"  # many_to_one, one_to_one, many_to_many


class SourceSchema(BaseModel):
    """Complete schema discovered from source database."""
    tables: list[TableSchema]
    relationships: list[Relationship] = Field(default_factory=list)
    estimated_total_rows: int = 0
    source_type: str = ""  # postgresql, mongodb, supabase, etc.
    source_version: Optional[str] = None


class ChangeEventType(str, Enum):
    """Type of change event."""
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"


class ChangeEvent(BaseModel):
    """Represents a change event for incremental sync."""
    event_type: ChangeEventType
    table_name: str
    primary_key: str
    data: Optional[dict] = None
    timestamp: datetime
    old_data: Optional[dict] = None  # For updates


class ConnectionConfig(BaseModel):
    """Configuration for database connection."""
    host: str
    port: int
    database: str
    username: str
    password: str
    ssl_mode: str = "prefer"
    extra_params: dict = Field(default_factory=dict)


class SourceAdapter(ABC):
    """Abstract base class for source database adapters.

    Implementations must provide methods for:
    - Connecting to the source database
    - Discovering the schema
    - Exporting data in batches
    - Tracking changes for incremental sync
    """

    @property
    @abstractmethod
    def adapter_type(self) -> str:
        """Return the adapter type identifier (e.g., 'postgresql', 'mongodb')."""
        pass

    @abstractmethod
    async def connect(self, config: ConnectionConfig) -> bool:
        """Establish connection to source database.

        Args:
            config: Connection configuration

        Returns:
            True if connection successful

        Raises:
            ConnectionError: If connection fails
        """
        pass

    @abstractmethod
    async def test_connection(self) -> bool:
        """Test if the connection is alive and valid.

        Returns:
            True if connection is healthy
        """
        pass

    @abstractmethod
    async def discover_schema(
        self,
        include_tables: Optional[list[str]] = None,
        exclude_tables: Optional[list[str]] = None,
    ) -> SourceSchema:
        """Auto-discover schema from source database.

        Args:
            include_tables: Whitelist of tables to include (None = all)
            exclude_tables: Blacklist of tables to exclude

        Returns:
            Discovered schema including tables, fields, and relationships
        """
        pass

    @abstractmethod
    async def estimate_size(self) -> dict[str, int]:
        """Estimate row counts per table for planning.

        Returns:
            Dict mapping table names to estimated row counts
        """
        pass

    @abstractmethod
    async def export_table(
        self,
        table_name: str,
        batch_size: int = 1000,
        cursor: Optional[str] = None,
        order_by: Optional[str] = None,
    ) -> AsyncIterator[tuple[list[dict], Optional[str]]]:
        """Stream table data in batches.

        Args:
            table_name: Name of the table to export
            batch_size: Number of records per batch
            cursor: Resume cursor from previous export
            order_by: Column to order by (default: primary key)

        Yields:
            Tuple of (batch of records, next cursor or None if complete)
        """
        pass

    @abstractmethod
    async def export_with_vectors(
        self,
        table_name: str,
        vector_column: str,
        batch_size: int = 1000,
        cursor: Optional[str] = None,
    ) -> AsyncIterator[tuple[list[dict], Optional[str]]]:
        """Stream table data with vector embeddings.

        Specifically for tables with pgvector or similar vector columns.

        Args:
            table_name: Name of the table to export
            vector_column: Name of the vector/embedding column
            batch_size: Number of records per batch
            cursor: Resume cursor from previous export

        Yields:
            Tuple of (batch of records with vectors, next cursor)
        """
        pass

    @abstractmethod
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
            Change events (inserts, updates, deletes)
        """
        pass

    @abstractmethod
    async def get_row_count(self, table_name: str) -> int:
        """Get exact row count for a table.

        Args:
            table_name: Name of the table

        Returns:
            Number of rows in the table
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Clean up connection resources."""
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
