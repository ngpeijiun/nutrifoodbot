"""I/O utilities for data pipeline.

Provides helpers to manage directories and DuckDB connections.
"""

from __future__ import annotations

from pathlib import Path
import duckdb  # type: ignore


def ensure_dir(path: Path | str) -> Path:
    """Ensure a directory exists and return it as Path.

    Parameters
    - path: Directory path to create if missing.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def default_db_path() -> Path:
    """Default location for the project DuckDB database file."""
    return Path("db") / "open_food.duckdb"


def connect_duckdb(db_path: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection to the provided path.

    If the path is None, uses the project default under `db/`.
    """
    path = Path(db_path) if db_path is not None else default_db_path()
    ensure_dir(path.parent)
    # Using read-write mode to create the file if it does not exist.
    conn = duckdb.connect(str(path))
    return conn
