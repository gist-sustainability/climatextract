# climatextract

climatextract is a retrieval-augmented generation (RAG) pipeline that surfaces CO₂ emissions data from corporate sustainability reports. It embeds PDF pages, ranks relevant context, and prompts a large language model to extract Scope 1-3 emissions into structured tables for downstream analysis.

## Background

This project began as the team's submission for the 2024 ClimateNLP workshop at ACL. Built by the [LMU SODA Lab](https://www.soda.statistik.uni-muenchen.de/) in collaboration with the [Data Service Centre of Deutsche Bundesbank](https://www.bundesbank.de/de/statistiken/nachhaltigkeit), climatextract combines research around ESG reporting and Intelligent Document Processing to automate what was previously a tedious manual annotation process.

This repository is organized as follows:

- `climatextract`: package source code
- `data`: source data to be analyzed
- `docs`: package documentation (built with mkdocs)
- `tests`: acceptance tests

## Setup

### Python environment

It is recommended to run the code in a virtual environment using at least Python 3.11.

First, check out the code, then create a virtual environment and install all dependencies:

```bash
cd climatextract
python -m venv co2_info_extraction
source co2_info_extraction/bin/activate
pip install -r requirements.txt
pip install -e .
```

### System dependencies

The Python package `pdf2image` is a wrapper around `poppler`, so you will need to install it as a system dependency.

**macOS:**
```bash
brew install poppler
```

**Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils
```

**Windows:** Download from the [@oschwartz10612 poppler build](https://github.com/oschwartz10612/poppler-windows/releases/), extract the archive, and add the `bin/` folder to your system's PATH environment variable. See the [pdf2image documentation](https://pypi.org/project/pdf2image/) for details.

### Azure OpenAI

climatextract uses Azure-hosted large language models for extraction and embedding. You will need to set up the following in your Azure account:

1. **A Large Language Model** accessible via Azure OpenAI or Azure AI Foundry. See the [configuration documentation](docs/user-guide/configuration.md) for supported models.
2. **An embedding model** accessible via Azure OpenAI (default: `text-embedding-ada-002`). Alternatively, you can use a local HuggingFace `sentence-transformers/*` model, which does not require Azure.

This package is currently tailored towards our Azure configuration and may not suit yours. Please check out the LLM class in [config.py](climatextract/config.py) to explore supported models and their parameters.

Create a `.env` file in the project root with your Azure credentials:

```bash
AZURE_ENDPOINT=https://your-endpoint.openai.azure.com/
API_KEY=your-api-key
API_VERSION=2024-12-01-preview
```

If your LLM is hosted via Azure AI Foundry (on a different endpoint), also add:

```bash
AZURE_AI_FOUNDRY_ENDPOINT=https://your-other-endpoint.openai.azure.com/
```

Instead of using an API key, you can also authenticate with your personal Azure account. See the [installation guide](docs/getting-started/installation.md) for details on personalized authentication via the [azure_authentication](https://github.com/soda-lmu/azure-auth-helper-python) package.

### MLflow experiment tracking (optional)

climatextract uses MLflow for experiment tracking. By default, experiments are tracked locally in a `./mlruns` directory. If you want to use a remote MLflow server (e.g. on Azure Databricks), add the following to your `.env` file:

```bash
MLFLOW_TRACKING_URI=databricks
DATABRICKS_HOST=https://your-databricks-instance.azuredatabricks.net
DATABRICKS_TOKEN=your-personal-access-token
```

See the [MLflow setup guide](docs/user-guide/mlflow-setup.md) for how to create a Databricks personal access token.

## Usage

Place your PDF sustainability reports in the `data/pdfs/` directory, then run the extraction pipeline:

```python
from climatextract import extract

result_path = extract("./data/pdfs/company_2023_report.pdf")
```

Results are saved as CSV files in `output/<run-id>/`. See the [Quickstart](docs/getting-started/quickstart.md) for more examples.

### Configuration

Extraction behavior is controlled via a `climatextract.toml` file in your working directory. It lets you configure the LLM model, embedding model, prompt type, year range, semantic search parameters, and more. See the [Configuration guide](docs/user-guide/configuration.md) for all available options.

### Running tests

```bash
python -m pytest
```

See `tests/README.md` for details on the acceptance test suite.

## Documentation

The full documentation covers usage, configuration, architecture, and API reference:

| Section | Description |
|---------|-------------|
| [Installation](docs/getting-started/installation.md) | Detailed setup instructions |
| [Quickstart](docs/getting-started/quickstart.md) | First extraction walkthrough |
| [Configuration](docs/user-guide/configuration.md) | All TOML configuration options |
| [Architecture](docs/concepts/architecture.md) | Pipeline design and components |
| [Prompts](docs/concepts/prompts.md) | How extraction prompts work |
| [Evaluation](docs/concepts/evaluation.md) | Measuring extraction quality |
| [API Reference](docs/api-reference/public-api.md) | Public API functions |

To build and serve the docs locally:

```bash
pip install -e '.[docs]'
mkdocs serve
```
