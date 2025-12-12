"""Field mapping engine for data migrations.

Maps source database fields to Vespa schema fields,
with support for type transformations and computed fields.
"""

import json
import logging
import re
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from .source_adapter import FieldSchema, FieldType

logger = logging.getLogger(__name__)


class VespaFieldType(str, Enum):
    """Vespa field types."""
    STRING = "string"
    INT = "int"
    LONG = "long"
    FLOAT = "float"
    DOUBLE = "double"
    BOOL = "bool"
    TENSOR = "tensor"
    ARRAY_STRING = "array<string>"
    ARRAY_INT = "array<int>"
    ARRAY_LONG = "array<long>"
    ARRAY_FLOAT = "array<float>"
    RAW = "raw"
    POSITION = "position"
    URI = "uri"


class TransformType(str, Enum):
    """Built-in transformation types."""
    NONE = "none"
    TRIM_WHITESPACE = "trim_whitespace"
    LOWERCASE = "lowercase"
    UPPERCASE = "uppercase"
    ISO_TO_EPOCH_MS = "iso_to_epoch_ms"
    EPOCH_TO_ISO = "epoch_to_iso"
    FLATTEN_JSON = "flatten_json"
    JSON_STRINGIFY = "json_stringify"
    JSON_PARSE = "json_parse"
    ARRAY_TO_STRING = "array_to_string"
    STRING_TO_ARRAY = "string_to_array"
    CAST_INT = "cast_int"
    CAST_FLOAT = "cast_float"
    CAST_STRING = "cast_string"
    CAST_BOOL = "cast_bool"
    EXTRACT_FIRST = "extract_first"
    COALESCE_EMPTY = "coalesce_empty"


class FieldMapping(BaseModel):
    """Configuration for mapping a single field."""
    source_field: str
    target_field: str
    transform: TransformType = TransformType.NONE
    custom_transform: Optional[str] = None  # Expression for custom transforms
    default_value: Optional[Any] = None
    is_required: bool = False
    is_embedding_source: bool = False  # Include in embedding generation


class ComputedField(BaseModel):
    """Configuration for a computed field."""
    name: str
    expression: str  # Python expression or SQL-like expression
    target_type: VespaFieldType = VespaFieldType.STRING


class EmbeddingConfig(BaseModel):
    """Configuration for embedding field generation."""
    source_fields: list[str]  # Fields to concatenate for embedding
    separator: str = " "
    target_field: str = "embedding"
    max_length: int = 8192  # Max chars before truncation


class TableMapping(BaseModel):
    """Complete mapping configuration for a table."""
    source_table: str
    target_schema: str  # Vespa schema name
    field_mappings: list[FieldMapping] = Field(default_factory=list)
    computed_fields: list[ComputedField] = Field(default_factory=list)
    embedding_config: Optional[EmbeddingConfig] = None
    primary_key_mapping: Optional[str] = None  # Source PK → Vespa doc ID


# Standard type mapping from source types to Vespa types
SOURCE_TO_VESPA_TYPE: dict[FieldType, VespaFieldType] = {
    FieldType.STRING: VespaFieldType.STRING,
    FieldType.TEXT: VespaFieldType.STRING,
    FieldType.INTEGER: VespaFieldType.INT,
    FieldType.BIGINT: VespaFieldType.LONG,
    FieldType.FLOAT: VespaFieldType.FLOAT,
    FieldType.DOUBLE: VespaFieldType.DOUBLE,
    FieldType.BOOLEAN: VespaFieldType.BOOL,
    FieldType.TIMESTAMP: VespaFieldType.LONG,  # Store as epoch ms
    FieldType.DATE: VespaFieldType.LONG,
    FieldType.JSON: VespaFieldType.STRING,  # Stored as JSON string
    FieldType.JSONB: VespaFieldType.STRING,
    FieldType.ARRAY: VespaFieldType.ARRAY_STRING,  # Default to string array
    FieldType.VECTOR: VespaFieldType.TENSOR,
    FieldType.UUID: VespaFieldType.STRING,
    FieldType.BINARY: VespaFieldType.RAW,
    FieldType.UNKNOWN: VespaFieldType.STRING,
}


class FieldMapper:
    """Engine for mapping and transforming fields during migration."""

    def __init__(self, table_mapping: TableMapping) -> None:
        """Initialize field mapper with configuration.

        Args:
            table_mapping: Table mapping configuration
        """
        self.mapping = table_mapping
        self._transform_funcs = self._build_transform_funcs()

    def _build_transform_funcs(self) -> dict[TransformType, Callable]:
        """Build dictionary of transform functions."""
        return {
            TransformType.NONE: lambda x: x,
            TransformType.TRIM_WHITESPACE: lambda x: x.strip() if isinstance(x, str) else x,
            TransformType.LOWERCASE: lambda x: x.lower() if isinstance(x, str) else x,
            TransformType.UPPERCASE: lambda x: x.upper() if isinstance(x, str) else x,
            TransformType.ISO_TO_EPOCH_MS: self._iso_to_epoch_ms,
            TransformType.EPOCH_TO_ISO: self._epoch_to_iso,
            TransformType.FLATTEN_JSON: self._flatten_json,
            TransformType.JSON_STRINGIFY: lambda x: json.dumps(x) if x else None,
            TransformType.JSON_PARSE: lambda x: json.loads(x) if isinstance(x, str) else x,
            TransformType.ARRAY_TO_STRING: lambda x: ",".join(str(i) for i in x) if isinstance(x, list) else x,
            TransformType.STRING_TO_ARRAY: lambda x: x.split(",") if isinstance(x, str) else x,
            TransformType.CAST_INT: lambda x: int(x) if x is not None else None,
            TransformType.CAST_FLOAT: lambda x: float(x) if x is not None else None,
            TransformType.CAST_STRING: lambda x: str(x) if x is not None else None,
            TransformType.CAST_BOOL: self._cast_bool,
            TransformType.EXTRACT_FIRST: lambda x: x[0] if isinstance(x, list) and x else x,
            TransformType.COALESCE_EMPTY: lambda x: x if x else "",
        }

    def _iso_to_epoch_ms(self, value: Any) -> Optional[int]:
        """Convert ISO timestamp to epoch milliseconds."""
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, datetime):
            return int(value.timestamp() * 1000)
        if isinstance(value, str):
            try:
                # Handle various ISO formats
                value = value.replace("Z", "+00:00")
                dt = datetime.fromisoformat(value)
                return int(dt.timestamp() * 1000)
            except Exception:
                return None
        return None

    def _epoch_to_iso(self, value: Any) -> Optional[str]:
        """Convert epoch milliseconds to ISO timestamp."""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        try:
            # Assume milliseconds
            if value > 1e12:
                value = value / 1000
            dt = datetime.fromtimestamp(value)
            return dt.isoformat()
        except Exception:
            return None

    def _flatten_json(self, value: Any) -> str:
        """Flatten nested JSON to dot-notation strings."""
        if value is None:
            return ""
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:
                return value

        if isinstance(value, dict):
            parts = []
            for k, v in value.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        parts.append(f"{k}.{k2}={v2}")
                else:
                    parts.append(f"{k}={v}")
            return "; ".join(parts)

        return str(value)

    def _cast_bool(self, value: Any) -> Optional[bool]:
        """Cast value to boolean."""
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)

    def transform_row(self, source_row: dict) -> dict:
        """Transform a source row to target format.

        Args:
            source_row: Row from source database

        Returns:
            Transformed row for Vespa ingestion
        """
        target_row = {}

        # Apply field mappings
        for field_map in self.mapping.field_mappings:
            source_value = source_row.get(field_map.source_field)

            # Apply default if source is None
            if source_value is None:
                if field_map.is_required:
                    raise ValueError(
                        f"Required field {field_map.source_field} is missing"
                    )
                source_value = field_map.default_value

            # Apply transformation
            transform_func = self._transform_funcs.get(
                field_map.transform,
                self._transform_funcs[TransformType.NONE]
            )

            try:
                target_value = transform_func(source_value)
            except Exception as e:
                logger.warning(
                    f"Transform failed for {field_map.source_field}: {e}"
                )
                target_value = field_map.default_value

            target_row[field_map.target_field] = target_value

        # Apply computed fields
        for computed in self.mapping.computed_fields:
            try:
                target_row[computed.name] = self._evaluate_expression(
                    computed.expression,
                    source_row,
                    target_row,
                )
            except Exception as e:
                logger.warning(f"Computed field {computed.name} failed: {e}")
                target_row[computed.name] = None

        return target_row

    def _evaluate_expression(
        self,
        expression: str,
        source_row: dict,
        target_row: dict,
    ) -> Any:
        """Evaluate computed field expression.

        Supports simple expressions like:
        - "value" - literal string
        - "${field}" - source field reference
        - "${source.field}" - explicit source reference
        - "${target.field}" - explicit target reference
        - "CASE WHEN ... THEN ... ELSE ... END" - simple case expression
        """
        # Handle literal values (quoted strings)
        if expression.startswith('"') and expression.endswith('"'):
            return expression[1:-1]
        if expression.startswith("'") and expression.endswith("'"):
            return expression[1:-1]

        # Handle field references
        def replace_ref(match):
            ref = match.group(1)
            if ref.startswith("source."):
                return str(source_row.get(ref[7:], ""))
            elif ref.startswith("target."):
                return str(target_row.get(ref[7:], ""))
            else:
                # Default to source
                return str(source_row.get(ref, ""))

        result = re.sub(r'\$\{([^}]+)\}', replace_ref, expression)

        # Handle simple CASE expressions
        case_match = re.match(
            r'CASE\s+WHEN\s+(.+?)\s+THEN\s+["\']?(.+?)["\']?\s+ELSE\s+["\']?(.+?)["\']?\s+END',
            result,
            re.IGNORECASE
        )
        if case_match:
            condition, then_val, else_val = case_match.groups()
            # Evaluate simple conditions
            if self._evaluate_condition(condition, source_row):
                return then_val
            return else_val

        return result

    def _evaluate_condition(self, condition: str, row: dict) -> bool:
        """Evaluate simple condition against row."""
        # Handle "field = value" patterns
        eq_match = re.match(r'(\w+)\s*=\s*["\']?(.+?)["\']?$', condition.strip())
        if eq_match:
            field, value = eq_match.groups()
            return str(row.get(field, "")) == value

        # Handle "field IS NULL" patterns
        null_match = re.match(r'(\w+)\s+IS\s+NULL', condition.strip(), re.IGNORECASE)
        if null_match:
            return row.get(null_match.group(1)) is None

        # Handle boolean field references
        if condition.strip() in row:
            return bool(row[condition.strip()])

        return False

    def get_embedding_text(self, source_row: dict) -> Optional[str]:
        """Extract text for embedding generation.

        Args:
            source_row: Row from source database

        Returns:
            Concatenated text for embedding, or None if no embedding config
        """
        if not self.mapping.embedding_config:
            return None

        config = self.mapping.embedding_config
        parts = []

        for field in config.source_fields:
            value = source_row.get(field)
            if value is not None:
                if isinstance(value, dict):
                    value = json.dumps(value)
                parts.append(str(value))

        text = config.separator.join(parts)

        # Truncate if needed
        if len(text) > config.max_length:
            text = text[:config.max_length]

        return text if text else None


def suggest_mappings(
    source_fields: list[FieldSchema],
    target_schema_name: str,
) -> TableMapping:
    """Generate suggested field mappings from source schema.

    Uses naming conventions and types to suggest appropriate mappings.

    Args:
        source_fields: List of source field schemas
        target_schema_name: Name of target Vespa schema

    Returns:
        Suggested TableMapping configuration
    """
    field_mappings = []
    embedding_source_fields = []

    for field in source_fields:
        target_name = field.name

        # Suggest transformation based on type
        transform = TransformType.NONE
        if field.type == FieldType.TIMESTAMP:
            transform = TransformType.ISO_TO_EPOCH_MS
        elif field.type in (FieldType.JSON, FieldType.JSONB):
            transform = TransformType.JSON_STRINGIFY

        # Identify likely embedding source fields
        is_embedding_source = False
        if field.name in ("name", "title", "description", "content", "summary", "text"):
            is_embedding_source = True
            embedding_source_fields.append(field.name)

        field_mappings.append(FieldMapping(
            source_field=field.name,
            target_field=target_name,
            transform=transform,
            is_embedding_source=is_embedding_source,
        ))

    # Create embedding config if we found suitable fields
    embedding_config = None
    if embedding_source_fields:
        embedding_config = EmbeddingConfig(
            source_fields=embedding_source_fields,
            separator=" ",
            target_field="embedding",
        )

    return TableMapping(
        source_table="",  # To be filled in
        target_schema=target_schema_name,
        field_mappings=field_mappings,
        embedding_config=embedding_config,
    )


def infer_vespa_type(field: FieldSchema) -> VespaFieldType:
    """Infer Vespa field type from source field schema.

    Args:
        field: Source field schema

    Returns:
        Appropriate Vespa field type
    """
    vespa_type = SOURCE_TO_VESPA_TYPE.get(field.type, VespaFieldType.STRING)

    # Special handling for vector fields
    if field.type == FieldType.VECTOR and field.vector_dimension:
        # Vespa tensor type depends on dimension
        return VespaFieldType.TENSOR

    # Handle arrays more specifically if we have sample values
    if field.type == FieldType.ARRAY and field.sample_values:
        sample = field.sample_values[0]
        if isinstance(sample, int):
            return VespaFieldType.ARRAY_INT
        elif isinstance(sample, float):
            return VespaFieldType.ARRAY_FLOAT

    return vespa_type
