from pipeline.jobs.base_ingest_job import BaseParquetToDuckDBJob

class IngestProductsJob(BaseParquetToDuckDBJob):
    """
    Job to ingest products data into DuckDB.
    """
    def __init__(self):
        super().__init__(
            db_path="db/open_food.duckdb",
            table_name="products",
            file_path="data_mining/data/product_features/*.parquet",
            columns=["code", "product_name", "image_key", "image_rev"],
        )

if __name__ == "__main__":
    job = IngestProductsJob()
    job.run()
