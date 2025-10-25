# NutriFoodBot

## Local Setup

1. Create and activate a Python virtual environment named `.venv`.
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install the project dependencies.
   ```bash
   pip install -r requirements.txt
   ```
3. Create `src/.env` with your OpenAI API key.
   ```bash
   echo "OPENAI_API_KEY=<your-key>" > src/.env
   ```

You can now run the project commands inside the virtual environment.

To launch the app, run:
```bash
streamlit run src/app.py
```

To run the full data pipeline, execute:
```bash
./run_pipeline.sh
```

If you need to rerun any pipeline steps, download the `food.parquet` file and place it in the `eda/` directory.
