from pipeline.jobs.base_ingest_job import BaseParquetToDuckDBJob

class IngestProductNutrientsJob(BaseParquetToDuckDBJob):
    """
    Job to ingest product nutrients data into DuckDB.
    """

    def __init__(self):
        super().__init__(
            db_path="db/open_food.duckdb",
            table_name="product_nutrients",
            file_path="data_mining/data/product_nutrients/*.parquet",
            exclude_columns=["__null_dask_index__"],
        )

if __name__ == "__main__":
    job = IngestProductNutrientsJob()
    job.run()
