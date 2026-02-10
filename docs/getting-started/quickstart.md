# Quickstart

Get up and running with climatextract in just a few minutes.

---

## Basic Extraction

The simplest way to extract emissions data from a PDF:

```python
from climatextract import extract

# Extract from a single PDF
result_path = extract("./reports/company_2023_report.pdf")
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
result_path = extract("./reports/")
```

Or provide a specific list:

```python
from climatextract import extract

files = [
    "./reports/apple_2021.pdf",
    "./reports/microsoft_2022.pdf",
]
result_path = extract(files)
```

---

## Extract and Evaluate

If you have a gold standard dataset, validate your results:

```python
from climatextract import extract_and_evaluate

result_path = extract_and_evaluate(
    pdf_input="./reports/",
    gold_standard_path="./evaluation/gold_standard.csv"
)
```

---

## Using a Configuration File

For more control, create a `climatextract.toml` configuration file:

```python
from climatextract import extract

# Uses settings from climatextract.toml
result_path = extract(config_path="climatextract.toml")
```

See [Configuration](../user-guide/configuration.md) for all available options.

---

## Output Structure

After extraction, you'll find results in `output/<run-id>/`:

```
output/
└── abc123-uuid/
    ├── 03_co2_emission_table2_w_query_responses.csv           # Page-level LLM responses (with duplicates)
    ├── 03_co2_emission_table2_w_query_responses_filtered.csv  # Deduplicated responses
    ├── intermediate_results.csv                               # Pre-normalization extraction results
    ├── results_long_format.csv                                # Main results in long format
    ├── results_wide_format.csv                                # Results pivoted by year
    ├── invalid_llm_outputs.txt                                # Invalid LLM responses
    └── logs.json                                              # Parameters, metrics, run info
```

See [Understanding Output](../user-guide/understanding-output.md) for column definitions.

---

## Next Steps

- [Configuration](../user-guide/configuration.md) – Customize extraction behavior
- [Architecture](../concepts/architecture.md) – Understand how the pipeline works
