#!/usr/bin/env python3
"""Execute all Jupyter notebooks in this directory in alphabetical order.

Usage:
  python data_mining/run_notebooks.py

The script finds all top-level `.ipynb` files under `data_mining/` (excluding
`.ipynb_checkpoints`), sorts them alphabetically, and executes each notebook
programmatically via nbconvert's ExecutePreprocessor, writing executed copies to
the `_executed/` subdirectory so original notebooks are not modified. It prints
per-notebook progress like "Executing cell X/Y...".

Notes:
- Ensure the virtual environment is active (e.g., `source proj_venv/bin/activate`).
- Requires Jupyter to be installed in the active environment.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbclient.exceptions import CellExecutionError


def list_notebooks(directory: Path) -> List[Path]:
    """Return a sorted list of top-level `.ipynb` files in `directory`.

    Excludes files inside `.ipynb_checkpoints` and hidden files.
    """
    notebooks: List[Path] = []
    for p in directory.iterdir():
        if not p.is_file():
            continue
        if p.suffix != ".ipynb":
            continue
        if p.name.startswith('.'):
            continue
        notebooks.append(p)
    notebooks.sort(key=lambda x: x.name)
    return notebooks


class ProgressExecutePreprocessor(ExecutePreprocessor):
    """Execute cells while printing progress like `X/Y` per notebook."""

    def __init__(self, total_code_cells: int, nb_name: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.total_code_cells = max(total_code_cells, 0)
        self.nb_name = nb_name
        self._executed = 0

    def preprocess_cell(self, cell, resources, cell_index):  # type: ignore[override]
        if getattr(cell, "cell_type", None) == "code":
            self._executed += 1
            total = self.total_code_cells or "?"
            print(f"[{self.nb_name}] Executing cell {self._executed}/{total}...")
        return super().preprocess_cell(cell, resources, cell_index)


def execute_notebook(
    nb_path: Path,
    output_dir: Path,
    timeout_seconds: int = 3600,
) -> int:
    """Execute a notebook and write the executed copy into `output_dir`.

    Prints per-cell progress (code cells only). Returns 0 on success, non-zero on failure.
    """
    print(f"\n=== Executing: {nb_path.name} ===")

    # Load notebook and count code cells for progress reporting.
    nb = nbformat.read(nb_path, as_version=4)
    total_code_cells = sum(1 for c in nb.cells if getattr(c, "cell_type", None) == "code")

    if total_code_cells == 0:
        print(f"[{nb_path.name}] No code cells found; copying to output without execution.")
        out_path = output_dir / nb_path.name
        nbformat.write(nb, out_path)
        return 0

    preproc = ProgressExecutePreprocessor(
        total_code_cells=total_code_cells,
        nb_name=nb_path.name,
        timeout=max(timeout_seconds, 600),  # cap per-notebook timeout to 10 min default
        kernel_name="python3",
    )

    # Ensure relative paths inside the notebook resolve from its directory.
    resources = {"metadata": {"path": str(nb_path.parent)}}

    try:
        preproc.preprocess(nb, resources)
    except CellExecutionError as exc:
        print(f"[{nb_path.name}] Execution failed: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 - show any unexpected error
        print(f"[{nb_path.name}] Unexpected error: {exc}")
        return 1

    # Write executed notebook to output directory with same filename.
    out_path = output_dir / nb_path.name
    nbformat.write(nb, out_path)
    print(f"[{nb_path.name}] Wrote executed notebook to {out_path}")
    return 0


def main() -> int:
    here = Path(__file__).resolve().parent
    output_dir = here / "_executed"
    output_dir.mkdir(parents=True, exist_ok=True)
    notebooks = list_notebooks(here)

    if not notebooks:
        print("No notebooks found to execute.")
        return 0

    print("Discovered notebooks (alphabetical):")
    for nb in notebooks:
        print(f" - {nb.name}")

    print(f"Output directory: {output_dir}")

    failures: List[str] = []
    for nb in notebooks:
        rc = execute_notebook(nb, output_dir=output_dir)
        if rc != 0:
            failures.append(f"{nb.name} (exit {rc})")

    if failures:
        print("\nSome notebooks failed:")
        for f in failures:
            print(f" - {f}")
        return 1

    print("\nAll notebooks executed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
