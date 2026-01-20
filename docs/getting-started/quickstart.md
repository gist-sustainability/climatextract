# Quickstart

Get up and running with climatextract in just a few minutes.

---

## Basic Extraction

The simplest way to extract emissions data from a PDF:

```python
from climatextract import extract

# Extract from a single PDF
result_path = extract("./data/pdfs/company_2023_report.pdf")
print(f"Results saved to: {result_path}")
```

This will:

1. Embed the PDF pages into a vector database
2. Search for pages relevant to CO₂ emissions
3. Use an LLM to extract Scope 1, 2, and 3 data
4. Save results to the `output/` directory

---

## Extract from Multiple PDFs

Process an entire directory of reports:

```python
from climatextract import extract

# Process all PDFs in a directory
result_path = extract("./data/pdfs/")
```

Or provide a specific list:

```python
from climatextract import extract

files = [
    "./data/pdfs/apple_2021_en.pdf",
    "./data/pdfs/allianz_2022_report.pdf",
]
result_path = extract(files)
```

---

## Extract and Evaluate

If you have a gold standard dataset, validate your results:

```python
from climatextract import extract_and_evaluate

result_path = extract_and_evaluate(
    pdf_input="./data/pdfs/sample_reports/",
    gold_standard_path="./data/evaluation_dataset/gist_2025.csv"
)
```

---

## Using a Configuration File

For more control, use the `climxtract.toml` configuration file:

```python
from climatextract import extract

# Uses settings from climxtract.toml
result_path = extract(config_path="climxtract.toml")
```

See [Configuration](../user-guide/configuration.md) for all available options.

---

## Output Structure

After extraction, you'll find results in `output/<run-id>/`:

```
output/
└── abc123-uuid/
    ├── results_long.csv      # Main results in long format
    ├── results_wide.csv      # Results pivoted by scope
    └── query_responses.csv   # Page-level LLM responses
```

See [Understanding Output](../user-guide/understanding-output.md) for column definitions.

---

## Next Steps

- [Configuration](../user-guide/configuration.md) – Customize extraction behavior
- [Architecture](../concepts/architecture.md) – Understand how the pipeline works
