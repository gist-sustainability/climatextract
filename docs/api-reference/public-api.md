# API Reference

This page documents the public API for climatextract. These are the only functions needed for typical usage.

---

## `extract`

Extract CO₂ emissions data from PDF reports.

```python
from climatextract import extract

result_path = extract(
    pdf_input="./data/pdfs/company_report.pdf",
    config_path="climatextract.toml",
    enable_mlflow=False
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdf_input` | `str \| List[str] \| None` | `None` | A directory path (processes all PDFs), a single file path, or a list of file paths. If `None`, uses `filename_list` from config. |
| `config_path` | `str` | `"climatextract.toml"` | Path to configuration file. |
| `enable_mlflow` | `bool` | `False` | Whether to log results to MLflow. If `True`, uses MLflow settings from config. |
| `verbose` | `bool` | `False` | Show detailed per-PDF output. |
| `llm` | `LlmHandler \| None` | `None` | Custom LLM handler (keyword-only). If `None`, builds the default Azure adapter from the `[models]` section of the config. See [Custom Providers](../user-guide/custom-providers.md). |
| `embedder` | `EmbeddingModelHandler \| None` | `None` | Custom embedding handler (keyword-only). If `None`, builds the default Azure adapter. See [Custom Providers](../user-guide/custom-providers.md). |

### Returns

| Type | Description |
|------|-------------|
| `str \| None` | Path to the results directory, or `None` if extraction failed. |

### Examples

**Single PDF:**

```python
result = extract("./data/pdfs/apple_2023.pdf")
```

**Directory of PDFs:**

```python
result = extract("./data/pdfs/sample_reports/")
```

**Multiple specific files:**

```python
result = extract([
    "./data/pdfs/apple_2023.pdf",
    "./data/pdfs/microsoft_2023.pdf"
])
```

**With MLflow tracking:**

```python
result = extract(
    pdf_input="./data/pdfs/",
    enable_mlflow=True
)
```

**With a custom provider handler:**

```python
from climatextract import extract
from climatextract.adapters.azure_openai import AzureOpenAILlmHandler

llm = AzureOpenAILlmHandler(model="gpt-5.2-chat-2025-12-11", reasoning_effort="low")
result = extract("./data/pdfs/", llm=llm)
```

---

## `extract_and_evaluate`

Extract CO₂ emissions data and evaluate against a gold standard dataset.

```python
from climatextract import extract_and_evaluate

result_path = extract_and_evaluate(
    pdf_input="./data/pdfs/sample_reports/",
    gold_standard_path="./data/evaluation_dataset/gist_2025.csv",
    config_path="climatextract.toml",
    enable_mlflow=False
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdf_input` | `str \| List[str] \| None` | `None` | A directory path, single file path, or list of file paths. If `None`, uses config. |
| `gold_standard_path` | `str \| None` | `None` | Path to gold standard CSV. If `None`, uses `gold_standard` from config. |
| `config_path` | `str` | `"climatextract.toml"` | Path to configuration file. |
| `enable_mlflow` | `bool` | `False` | Whether to log results and metrics to MLflow. |
| `verbose` | `bool` | `False` | Show detailed per-PDF output. |
| `llm` | `LlmHandler \| None` | `None` | Custom LLM handler (keyword-only). If `None`, builds the default Azure adapter from the `[models]` section of the config. See [Custom Providers](../user-guide/custom-providers.md). |
| `embedder` | `EmbeddingModelHandler \| None` | `None` | Custom embedding handler (keyword-only). If `None`, builds the default Azure adapter. See [Custom Providers](../user-guide/custom-providers.md). |

### Returns

| Type | Description |
|------|-------------|
| `str \| None` | Path to the results directory (includes evaluation files), or `None` if failed. |

### Examples

**Basic evaluation:**

```python
result = extract_and_evaluate(
    pdf_input="./data/pdfs/test_set/",
    gold_standard_path="./data/evaluation_dataset/gist_2025.csv"
)
```

**Using config defaults:**

```python
# Uses pdf_input and gold_standard from climatextract.toml
result = extract_and_evaluate()
```

**With MLflow tracking:**

```python
result = extract_and_evaluate(
    pdf_input="./data/pdfs/",
    gold_standard_path="./data/evaluation/gold.csv",
    enable_mlflow=True
)
```

---

## Import

Both functions are available directly from the `climatextract` package:

```python
from climatextract import extract, extract_and_evaluate
```

---

## Next Steps

- [Configuration](../user-guide/configuration.md) – Customize extraction behavior
- [Background](../research/background.md) – Academic context and motivation
