# Running Extraction

This guide covers different ways to run climatextract on your PDF reports.

---

## Using Python API

### Single PDF

```python
from climatextract import extract

result_path = extract("./reports/company_2023_report.pdf")
```

### Multiple PDFs

```python
from climatextract import extract

files = [
    "./reports/apple_2021.pdf",
    "./reports/allianz_2022.pdf",
]
result_path = extract(files)
```

### Directory of PDFs

```python
from climatextract import extract

# Processes all .pdf files in the directory
result_path = extract("./reports/")
```

---

## Using Configuration File

For reproducible runs, create a `climatextract.toml` configuration file:

```python
from climatextract import extract

# All settings from climatextract.toml
result_path = extract(config_path="climatextract.toml")
```

Override specific inputs while using config defaults:

```python
from climatextract import extract

result_path = extract(
    pdf_input="./reports/new_report.pdf",
    config_path="climatextract.toml"
)
```

See [Configuration](configuration.md) for all available options.

---

## Extract with Evaluation

Compare results against a gold standard dataset:

```python
from climatextract import extract_and_evaluate

result_path = extract_and_evaluate(
    pdf_input="./reports/",
    gold_standard_path="./evaluation/gold_standard.csv"
)
```

!!! note "Evaluation Output"
    Evaluation adds additional files to the output directory with precision, recall, and F1 scores.

---

## MLflow Tracking

Enable experiment tracking by setting `enable_mlflow=True`:

```python
from climatextract import extract

result_path = extract(
    pdf_input="./reports/",
    enable_mlflow=True
)
```

When enabled, metrics and parameters are logged to your configured MLflow server.

---

## Processing Tips

!!! tip "Large Batches"
    When processing many PDFs, consider:
    
    - Reducing `max_parallel_llm_prompts_running` to avoid rate limits
    - Using a pre-built embeddings database to skip re-embedding
    - Monitoring the `output/` directory for intermediate results

!!! warning "Memory Usage"
    Large PDF files can consume significant memory during embedding. Process in smaller batches if you encounter memory issues.
