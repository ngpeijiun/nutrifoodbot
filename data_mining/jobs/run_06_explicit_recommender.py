"""Job for executing the explicit recommender notebook."""

from __future__ import annotations

from pathlib import Path

from data_mining.run_notebooks import execute_notebook


class RunExplicitRecommenderJob:
    """Execute `06_explicit_recommender.ipynb` via nbconvert."""

    def __init__(
        self,
        notebook_path: str | Path = "data_mining/06_explicit_recommender.ipynb",
        output_dir: str | Path = "data_mining/_executed",
        timeout_seconds: int = 3600,
    ) -> None:
        self.notebook_path = Path(notebook_path)
        self.output_dir = Path(output_dir)
        self.timeout_seconds = timeout_seconds

    def run(self) -> None:
        """Execute the notebook and persist the executed copy."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return_code = execute_notebook(
            nb_path=self.notebook_path,
            output_dir=self.output_dir,
            timeout_seconds=self.timeout_seconds,
        )
        if return_code != 0:
            raise RuntimeError(
                "Notebook execution failed with exit code "
                f"{return_code} for {self.notebook_path.name}"
            )


if __name__ == "__main__":
    job = RunExplicitRecommenderJob()
    job.run()
