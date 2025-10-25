from pathlib import Path
import glob

import duckdb
import pandas as pd

class BaseParquetToDuckDBJob:
    """
    Base class for ingesting Parquet files into DuckDB.
    """

    def __init__(self, db_path: str, table_name: str, file_path: str, columns: list[str] = None, exclude_columns: list[str] = None):
        self.db_path = Path(db_path)
        self.table_name = table_name
        self.file_path = Path(file_path)
        self.columns = columns
        self.exclude_columns = exclude_columns

    def run(self):
        """
        Validate inputs, load Parquet data, and ingest into DuckDB.
        """
        self._validate_inputs()

        if self.columns and self.exclude_columns:
            raise ValueError("Cannot use both 'columns' and 'exclude_columns'. Use only one.")

        if self.exclude_columns:
            select_clause = "*"
            exclude_clause = f" EXCLUDE ({', '.join(self.exclude_columns)})"
        elif self.columns:
            select_clause = ", ".join(self.columns)
            exclude_clause = ""
        else:
            select_clause = "*"
            exclude_clause = ""

        with duckdb.connect(self.db_path.as_posix()) as conn:
            query = f"""
            CREATE OR REPLACE TABLE {self.table_name} AS
            SELECT {select_clause}{exclude_clause}
            FROM read_parquet('{self.file_path.as_posix()}')
            """
            conn.execute(query)

    def _validate_inputs(self):
        """
        Ensure file and database paths exist.
        """
        # Check if the file path contains a wildcard
        if "*" in self.file_path.as_posix():
            matching_files = glob.glob(self.file_path.as_posix())
            if not matching_files:
                raise FileNotFoundError(f"No files found matching: {self.file_path}")
        else:
            if not self.file_path.exists():
                raise FileNotFoundError(f"File not found: {self.file_path}")
