# climatextract

📖 **Documentation:** [gist-sustainability.github.io/climatextract](https://gist-sustainability.github.io/climatextract/)

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

See the [Installation guide](https://gist-sustainability.github.io/climatextract/getting-started/installation/) for additional steps and alternative deployment options you have.

## Usage

Place your PDF sustainability reports in the `data/pdfs/` directory, then run the extraction pipeline:

```python
from climatextract import extract

result_path = extract("./data/pdfs/company_2023_report.pdf")
```

Results are saved as CSV files in `output/<run-id>/`. See the [Quickstart](https://gist-sustainability.github.io/climatextract/getting-started/quickstart/) for more examples.

### Configuration

Extraction behavior is controlled via a `climatextract.toml` file in your working directory. It lets you configure the LLM model, embedding model, prompt type, year range, semantic search parameters, and more. See the [Configuration guide](https://gist-sustainability.github.io/climatextract/user-guide/configuration/) for all available options.

### Running tests

```bash
python -m pytest
```

See `tests/README.md` for details on the acceptance test suite.

## Documentation

The full documentation is hosted at [gist-sustainability.github.io/climatextract](https://gist-sustainability.github.io/climatextract/) and covers usage, configuration, architecture, and API reference:

| Section | Description |
|---------|-------------|
| [Installation](https://gist-sustainability.github.io/climatextract/getting-started/installation/) | Detailed setup instructions |
| [Quickstart](https://gist-sustainability.github.io/climatextract/getting-started/quickstart/) | First extraction walkthrough |
| [Configuration](https://gist-sustainability.github.io/climatextract/user-guide/configuration/) | All TOML configuration options |
| [Custom Providers](https://gist-sustainability.github.io/climatextract/user-guide/custom-providers/) | Plug in a non-Azure LLM or embedding backend |
| [Architecture](https://gist-sustainability.github.io/climatextract/concepts/architecture/) | Pipeline design and components |
| [Prompts](https://gist-sustainability.github.io/climatextract/concepts/prompts/) | How extraction prompts work |
| [Evaluation](https://gist-sustainability.github.io/climatextract/concepts/evaluation/) | Measuring extraction quality |
| [API Reference](https://gist-sustainability.github.io/climatextract/api-reference/public-api/) | Public API functions |

To build and serve the docs locally:

```bash
pip install -e '.[docs]'
mkdocs serve
```
