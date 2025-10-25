from pipeline.jobs.base_ingest_job import BaseParquetToDuckDBJob

class IngestProductNutrientProfilesJob(BaseParquetToDuckDBJob):
    """
    Job to ingest product nutrient profiles data into DuckDB.
    """

    def __init__(self):
        super().__init__(
            db_path="db/open_food.duckdb",
            table_name="product_nutrient_profiles",
            file_path="data_mining/data/product_nutrient_profiles/*.parquet",
            exclude_columns=["__null_dask_index__"],
        )

if __name__ == "__main__":
    job = IngestProductNutrientProfilesJob()
    job.run()
