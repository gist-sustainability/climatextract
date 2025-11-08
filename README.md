# information-extraction-pilot

Information-extraction-pilot is a retrieval-augmented generation (RAG) pipeline that surfaces CO₂ emissions data from corporate sustainability reports. It embeds PDF pages, ranks relevant context, and prompts a large language model to extract Scope 1–3 emissions into structured tables for downstream analysis.

## Background

This pilot began as the team’s submission for the 2024 ClimateNLP workshop at ACL. The repository now serves as the maintained codebase for automating emissions extraction, while retaining the project’s research lineage.

This repository is organized as follows:

- `data`: source data to be analyzed and the gold standard dataset
- `output`: pipeline results
- `prompt`: prompt templates and queries
- `src`: pipeline source code
- `tests`: automated checks for the pilot

# Setup

## Python environment

It is recommended to run the code in a virtual environment using at least Python 3.11:

If you are using `pip`, run

`python3.11 -m venv co2_info_extraction` `pip install -r requirements.txt`

to install all dependencies.

## Other dependencies

Since the python package `pdf2image` is a wrapper around `poppler`, you will need to install it. See https://pypi.org/project/pdf2image/

## Azure Authentication

This repository uses Azure modules, so you need to have access to it. The code relies on the presence of an `.env` file that stores your credentials. Configure your own authentication workflow with environment variables, see the [description](https://github.com/soda-lmu/azure-auth-helper-python/blob/main/AuthenticationWorkflowSetup.md).

## Azure Databricks

Furthermore, the repository uses `mlflow` for tracking of experiments. To set up access to the Mlflow Tracking Server on Azure Databricks, you need to create a personal access token. Follow the following steps: 

1. Log into [Azure](https://portal.azure.com). 
2. Search for `gist-mlflow-tracking-server` to find the respective Databricks instance. 
3. Copy the URL which contains azuredatabricks.net and save it in the `.env` file as `DATABRICKS_HOST` variable. 
4. Save the variable `MLFLOW_TRACKING_URI` with the value `databricks` to the `.env` file.  
5. Launch the workspace and click on your initial in the upper right corner. 
6. Navigate to `Settings > User > Developer > Access tokens`and click on `Manage`. Generate a new access token and save it in the `.env` file as `DATABRICKS_TOKEN` variable. Be aware that it takes some time for the token to get activated, so you might get 401 authentication errors in the beginning when running the code. This should be resolved after some time.

## Run of main.py

The script uses three dataclasses to manage configurations: `MlflowParams`, `ConfigParams`, and `ExperimentParams`. These can be customized directly in `main.py` or through external configuration files integrated into `config.py`.

### Key Parameters

Parameters that can be updated through the `helpers.update_dataclass()` function.

**ConfigParams:**

- `gold_standard`: Currently supports `gist_2025` (default)

- `filename_list`: List of filenames that will be input into the pipeline, can be adjusted manually or via the function `helpers.get_file_paths`

**ExperimentParams:**

-   `emb_model`: Name of the embedding model.

-   `llm_model`: Name of the LLM to use.

-   `prompt_type`: Type of prompt (default or custom_gaia).

-   `search_query`: Query passed to the pipeline.

-   `year_min` and `year_max`: Filters for data based on year.

## Running the Script

### Standard Execution

To run the pipeline, execute:

`python main.py`

### Customizing Parameters

Modify the parameters in `main.py` by updating the relevant dataclass instances. For example:

`helpers.update_dataclass(config_params, {      'filename_list': ['./data/pdfs/apple_2021_en.pdf'], }) helpers.update_dataclass(experiment_params, {     'prompt_type': 'custom_gaia',     'search_query': "What are the carbon emissions for the last 10 years?", })`

### Logging and Debugging

-   Set the desired log level in the `logging.basicConfig()` call, e.g., `logging.DEBUG` for verbose logs.

-   Outputs and errors will appear in the console.
