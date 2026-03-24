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

climatextract uses Azure-hosted large language models for extraction and embedding. You will need an LLM and an embedding model accessible via Azure OpenAI or Azure AI Foundry.

See the [Installation guide](docs/getting-started/installation.md) for how to configure your `.env` file with Azure credentials and authentication options.

### MLflow experiment tracking (optional)

climatextract uses MLflow for experiment tracking. By default, experiments are tracked locally in a `./mlruns` directory.

To set up remote tracking via Azure Databricks, see the [MLflow setup guide](docs/user-guide/mlflow-setup.md).

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
