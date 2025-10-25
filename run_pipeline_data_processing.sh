#!/usr/bin/env bash

python -m data_mining.jobs.run_01_dataset_foundation_analysis
python -m data_mining.jobs.run_02_nutritional_composition_analysis
python -m data_mining.jobs.run_03_ingredient_analysis_substitution
python -m data_mining.jobs.run_04_food_classification_analysis
