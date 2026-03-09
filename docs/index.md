# climatextract

**Extract CO₂ emissions data from corporate sustainability reports using AI.**

climatextract is an information extraction pipeline that surfaces Scope 1, 2, and 3 emissions data from PDF sustainability reports. Built by the [LMU SODA Lab](https://www.soda.statistik.uni-muenchen.de/) in collaboration with the [Data Service Centre of Deutsche Bundesbank](https://www.bundesbank.de/de/statistiken/nachhaltigkeit), it combines research around ESG reporting and Intelligent Document Processing. Employing semantic search and large language models, climatextract automates what was previously a tedious manual annotation process.



---

## Key Features

- 📄 **PDF Processing** – Automatically extract and embed text from sustainability reports
- 🔍 **Semantic Search** – Find relevant pages using vector similarity
- 🤖 **LLM Extraction** – Use GPT models to extract structured emissions data
- 📊 **Scope 1-3 Coverage** – Extract direct and indirect emissions across all scopes
- ✅ **Evaluation** – Compare results against gold standard datasets

---

## Quick Example

```python
from climatextract import extract

# Extract emissions from a PDF report
result_path = extract("./data/pdfs/company_2023_report.pdf")
print(f"Results saved to: {result_path}")
```

The main output is a well structured table, saved in .csv format.

| report_id | year | indicator | value_std | ... | unit_std | ... | page |
|-----------|------|-----------|-----------|-----|----------|-----|------|
| company_2023_report.pdf | 2015 | scope 1 | 135.0 | ... | t CO2e | ... | 34 |
| company_2023_report.pdf | 2015 | scope 2lb | 41962.0 | ... | t CO2e | ... | 34 |
| company_2023_report.pdf | 2015 | scope 2mb | 37674.0 | ... | t CO2e | ... | 34 |
| company_2023_report.pdf | 2015 | scope 3 | 1834.0 | ... | t CO2e | ... | 34 |
| company_2023_report.pdf | 2016 | scope 1 | 170.0 | ... | t CO2e | ... | 34 |
| ... | ... | ... | ... | ... | ... | ... | ... |

---

## Getting Started

New to climatextract? Start here:

- [**Installation**](getting-started/installation.md) – Set up your environment
- [**Quickstart**](getting-started/quickstart.md) – Run your first extraction

---

## Documentation Overview

| Section | Description |
|---------|-------------|
| [**User Guide**](user-guide/configuration.md) | Configuration, running extractions, understanding output |
| [**Concepts**](concepts/architecture.md) | Architecture, RAG pipeline, prompts, evaluation |
| [**API Reference**](api-reference/public-api.md) | Public API functions |
| [**Research**](research/background.md) | Academic background, methodology, citation |
