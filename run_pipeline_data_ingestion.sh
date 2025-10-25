#!/usr/bin/env bash

python -m pipeline.jobs.ingest_products
python -m pipeline.jobs.ingest_product_categories
python -m pipeline.jobs.ingest_product_nutrients
python -m pipeline.jobs.ingest_product_nutrient_profiles
python -m pipeline.jobs.ingest_product_ingredients
