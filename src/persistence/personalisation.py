"""Persistence helpers for personalisation metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb

DB_PATH = Path(__file__).resolve().parents[2] / "db" / "open_food.duckdb"

_DIETARY_COLUMNS_SQL = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = 'main'
      AND table_name = 'product_categories'
      AND column_name LIKE 'is_%_compatible'
    ORDER BY column_name
"""

_ALLERGEN_COLUMNS_SQL = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = 'main'
      AND table_name = 'product_ingredients'
      AND column_name LIKE 'contains_%'
    ORDER BY column_name
"""


class PersonalisationStore:
    """Load personalisation metadata from DuckDB."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)

    def _connect(self, read_only: bool = False) -> duckdb.DuckDBPyConnection:
        """Return a DuckDB connection with the requested access mode."""

        return duckdb.connect(self._db_path.as_posix(), read_only=read_only)

    def list_dietary_columns(self) -> list[str]:
        """Return dietary compatibility column names."""

        try:
            with self._connect(read_only=True) as conn:
                rows = conn.execute(_DIETARY_COLUMNS_SQL).fetchall()
        except duckdb.Error:
            return []
        return [column for (column,) in rows]

    def list_allergen_columns(self) -> list[str]:
        """Return allergen indicator column names."""

        try:
            with self._connect(read_only=True) as conn:
                rows = conn.execute(_ALLERGEN_COLUMNS_SQL).fetchall()
        except duckdb.Error:
            return []
        return [column for (column,) in rows]

    def load_user_profile(self, user_id: str) -> dict[str, Any] | None:
        """Return the saved profile for ``user_id`` if available."""

        query = "SELECT profile FROM user_profiles WHERE user_id = ?"
        try:
            with self._connect(read_only=False) as conn:
                conn.execute(_CREATE_PROFILE_TABLE_SQL)
                result = conn.execute(query, [user_id]).fetchone()
        except duckdb.Error:
            return None
        if result is None:
            return None
        profile_json: str = result[0]
        try:
            data: dict[str, Any] = json.loads(profile_json)
        except json.JSONDecodeError:
            return None
        return data

    def save_user_profile(self, user_id: str, profile: dict[str, Any]) -> bool:
        """Persist ``profile`` for ``user_id`` in DuckDB."""

        payload = json.dumps(profile)
        upsert_sql = """
            INSERT INTO user_profiles (user_id, profile)
            VALUES (?, ?)
            ON CONFLICT (user_id) DO UPDATE SET profile = excluded.profile
        """
        try:
            with self._connect(read_only=False) as conn:
                conn.execute(_CREATE_PROFILE_TABLE_SQL)
                conn.execute(upsert_sql, [user_id, payload])
        except duckdb.Error:
            return False
        return True


_CREATE_PROFILE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS user_profiles (
        user_id VARCHAR PRIMARY KEY,
        profile JSON
    )
"""


PERSONALISATION_STORE = PersonalisationStore(DB_PATH)

__all__ = ["PersonalisationStore", "PERSONALISATION_STORE"]
