from pipeline.jobs.base_ingest_job import BaseParquetToDuckDBJob

class IngestProductCategoriesJob(BaseParquetToDuckDBJob):
    """
    Job to ingest product categories data into DuckDB.
    """

    def __init__(self):
        super().__init__(
            db_path="db/open_food.duckdb",
            table_name="product_categories",
            file_path="data_mining/data/product_features/*.parquet",
            exclude_columns=[
                "product_name",
                "image_key",
                "image_rev",
                "__null_dask_index__",
            ],
        )

if __name__ == "__main__":
    job = IngestProductCategoriesJob()
    job.run()
