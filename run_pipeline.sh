#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/run_pipeline_data_processing.sh"
bash "${SCRIPT_DIR}/run_pipeline_data_ingestion.sh"
bash "${SCRIPT_DIR}/run_pipeline_machine_learning.sh"
bash "${SCRIPT_DIR}/run_pipeline_deploy_model.sh"
