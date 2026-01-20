# Background

climatextract is a research project developed at the [LMU SODA Lab](https://www.soda.statistik.uni-muenchen.de/) (Statistics and Data Science in Social Sciences and the Humanities) at Ludwig-Maximilians-Universität München.

---

## Origin

The project began as the team's submission for the **ClimateNLP Workshop at ACL 2024**. This workshop focuses on Natural Language Processing techniques applied to climate-related text, including:

- Climate policy analysis
- Sustainability report processing
- Environmental data extraction

---

## Problem Statement

Corporate sustainability reports contain valuable emissions data, but extracting this information manually is:

- **Time-consuming** – Reports can be 100+ pages
- **Inconsistent** – Data formats vary across companies
- **Error-prone** – Manual transcription introduces mistakes
- **Unscalable** – Thousands of reports are published annually

climatextract automates this process using modern NLP techniques.

---

## Research Goals

The project aims to:

1. **Demonstrate feasibility** of automated emissions extraction
2. **Benchmark performance** against human annotations
3. **Explore RAG architectures** for structured data extraction
4. **Enable large-scale analysis** of corporate emissions trends

---

## Institutional Context

The project is part of GIST (Green IT and Sustainability Tracking) research efforts at LMU Munich, contributing to:

- Academic research on climate disclosure analysis
- Open-source tools for sustainability researchers
- Methodological advances in document AI

---

## Related Work

climatextract builds on advances in:

- **Retrieval-Augmented Generation** – Combining search with LLMs
- **Document Understanding** – Extracting structured data from PDFs
- **Climate NLP** – Applying NLP to environmental text

For technical details, see [Architecture](../concepts/architecture.md) and [Methodology](methodology.md).
