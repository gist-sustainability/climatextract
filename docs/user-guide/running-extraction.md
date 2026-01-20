# Running Extraction

This guide covers different ways to run climatextract on your PDF reports.

---

## Using Python API

### Single PDF

```python
from climatextract import extract

result_path = extract("./data/pdfs/company_2023_report.pdf")
```

### Multiple PDFs

```python
from climatextract import extract

files = [
    "./data/pdfs/apple_2021_en.pdf",
    "./data/pdfs/allianz_2022_report.pdf",
]
result_path = extract(files)
```

### Directory of PDFs

```python
from climatextract import extract

# Processes all .pdf files in the directory
result_path = extract("./data/pdfs/sample_reports/")
```

---

## Using Configuration File

For reproducible runs, use the configuration file:

```python
from climatextract import extract

# All settings from climxtract.toml
result_path = extract(config_path="climxtract.toml")
```

Override specific inputs while using config defaults:

```python
from climatextract import extract

result_path = extract(
    pdf_input="./data/pdfs/new_report.pdf",
    config_path="climxtract.toml"
)
```

---

## Extract with Evaluation

Compare results against a gold standard dataset:

```python
from climatextract import extract_and_evaluate

result_path = extract_and_evaluate(
    pdf_input="./data/pdfs/sample_reports/",
    gold_standard_path="./data/evaluation_dataset/gist_2025.csv"
)
```

!!! note "Evaluation Output"
    Evaluation adds additional files to the output directory with precision, recall, and F1 scores.

---

## Command Line Usage

You can also run extraction from the command line:

```bash
python -m climatextract.main
```

This uses settings from `climxtract.toml` by default.

---

## MLflow Tracking

Enable experiment tracking:

```python
from climatextract.main import main

# With MLflow tracking
main(use_mlflow=True)

# Without MLflow tracking  
main(use_mlflow=False)
```

---

## Processing Tips

!!! tip "Large Batches"
    When processing many PDFs, consider:
    
    - Reducing `max_parallel_llm_prompts_running` to avoid rate limits
    - Using a pre-built embeddings database to skip re-embedding
    - Monitoring the `output/` directory for intermediate results

!!! warning "Memory Usage"
    Large PDF files can consume significant memory during embedding. Process in smaller batches if you encounter memory issues.
