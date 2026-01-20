# climatextract

**Extract CO₂ emissions data from corporate sustainability reports using AI.**

climatextract is a Retrieval-Augmented Generation (RAG) pipeline that surfaces Scope 1, 2, and 3 emissions data from PDF sustainability reports. Built by the [LMU SODA Lab](https://www.soda.statistik.uni-muenchen.de/), it combines semantic search with large language models to automate what was previously a tedious manual process.

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
