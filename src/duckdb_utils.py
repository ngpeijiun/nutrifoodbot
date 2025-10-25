"""Utilities for working with DuckDB identifiers and type metadata."""

from __future__ import annotations


def quote_identifier(identifier: str) -> str:
    """Escape an identifier for use in DuckDB SQL statements."""

    return '"' + identifier.replace('"', '""') + '"'


def qualified_table_name(schema: str, table: str) -> str:
    """Return a fully qualified table name if the schema is not implicit."""

    if schema.lower() in {"main", "temp"}:
        return quote_identifier(table)
    return f"{quote_identifier(schema)}.{quote_identifier(table)}"


def display_table_name(schema: str, table: str) -> str:
    """Return a human-readable table name with schema when necessary."""

    if schema.lower() in {"main", "temp"}:
        return table
    return f"{schema}.{table}"


def is_numeric_type(type_str: str) -> bool:
    """Decide whether a DuckDB type string represents a numeric column."""

    t = type_str.lower()
    numeric_prefixes = (
        "tinyint",
        "smallint",
        "integer",
        "bigint",
        "hugeint",
        "utinyint",
        "usmallint",
        "uinteger",
        "ubigint",
        "real",
        "float",
        "double",
        "decimal",
        "numeric",
    )
    return t.startswith(numeric_prefixes)
