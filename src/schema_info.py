"""DuckDB schema inspection helpers."""

from __future__ import annotations

import duckdb
from duckdb_utils import (
    display_table_name,
    is_numeric_type,
    qualified_table_name,
    quote_identifier,
)


def process_column(
    conn: duckdb.DuckDBPyConnection,
    lines: list[str],
    low_cardinality: int,
    table_ref: str,
    col_name: str,
    col_type: str,
) -> None:
    """Append column statistics to the provided lines list."""

    column_header = f"- {col_name} ({col_type})"
    col_ref = quote_identifier(col_name)
    lines.append(column_header)

    if is_numeric_type(col_type):
        min_v, median_v, mean_v, max_v = conn.execute(
            f"""
            SELECT
                MIN({col_ref}),
                MEDIAN({col_ref}),
                AVG({col_ref}),
                MAX({col_ref})
            FROM {table_ref}
            """
        ).fetchone()
        stats = {
            "min": min_v,
            "median": median_v,
            "mean": mean_v,
            "max": max_v,
        }
        formatted = ", ".join(
            f"{k}={('NULL' if v is None else format(v, '.6g'))}"
            for k, v in stats.items()
        )
        lines.append(f"  - numeric stats: {formatted}")
        return

    (distinct_count,) = conn.execute(
        f"SELECT COUNT(DISTINCT {col_ref}) FROM {table_ref}"
    ).fetchone()
    if distinct_count <= low_cardinality:
        values = conn.execute(
            f"""
            SELECT DISTINCT {col_ref}
            FROM {table_ref}
            ORDER BY {col_ref} NULLS LAST
            LIMIT {low_cardinality}
            """
        ).fetchall()
        pretty = ["NULL" if v[0] is None else str(v[0]) for v in values]
        lines.append(f"  - distinct values ({len(pretty)}): {pretty}")
        return

    lines.append(f"  - high cardinality: {distinct_count} distinct")


def process_table(
    conn: duckdb.DuckDBPyConnection,
    lines: list[str],
    low_cardinality: int,
    schema_name: str | None,
    table_name: str,
) -> None:
    """Append table statistics to the provided lines list."""

    schema_label = schema_name or "main"
    table_ref = qualified_table_name(schema_label, table_name)
    display_name = display_table_name(schema_label, table_name)

    row_count = conn.execute(f"SELECT COUNT(*) FROM {table_ref}").fetchone()[0]

    lines.append(f"Table: {display_name} (rows: {row_count})")
    lines.append("Columns and stats:")

    columns = conn.execute(f"PRAGMA table_info({table_ref})").fetchall()

    for _, col_name, col_type, *_ in columns:
        process_column(conn, lines, low_cardinality, table_ref, col_name, col_type)


def get_schema_info(db_path: str) -> str:
    """Generate a schema overview and column statistics for all tables in a DuckDB.

    This function uses DuckDB introspection to list table columns and their data
    types. It also computes column-level statistics:
      - Numeric columns: min, median, mean, and max values.
      - Non-numeric columns: distinct values if cardinality is <= 20; otherwise,
        the count of distinct values is provided.

    Args:
        db_path (str): Path to the DuckDB database file.

    Returns:
        str: A human-readable summary of the schema and column statistics.
    """

    try:
        with duckdb.connect(db_path, read_only=True) as conn:
            tables = conn.execute(
                """
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_schema NOT IN ('information_schema')
                ORDER BY table_schema, table_name
                """
            ).fetchall()

            if not tables:
                return f"No tables found in {db_path}."

            lines: list[str] = []
            low_cardinality = 20

            for schema, table in tables:
                process_table(conn, lines, low_cardinality, schema, table)

            return "\n".join(lines)
    except duckdb.Error as exc:  # pragma: no cover - protective branch
        return f"Failed to connect to {db_path}: {exc}"
