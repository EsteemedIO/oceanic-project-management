"""Migration service package for Oceanic data migrations.

Provides adapters for migrating data from various sources to Vespa.
"""

from .source_adapter import SourceAdapter, SourceSchema, TableSchema, FieldSchema, ChangeEvent
from .adapters.postgresql_adapter import PostgreSQLAdapter
from .adapters.supabase_adapter import SupabaseAdapter

__all__ = [
    "SourceAdapter",
    "SourceSchema",
    "TableSchema",
    "FieldSchema",
    "ChangeEvent",
    "PostgreSQLAdapter",
    "SupabaseAdapter",
]
