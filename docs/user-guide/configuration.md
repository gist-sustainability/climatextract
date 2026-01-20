# Configuration

climatextract uses a TOML configuration file (`climxtract.toml`) to control all aspects of extraction. This guide explains each option.

---

## Configuration File Location

By default, the pipeline looks for `climxtract.toml` in the project root. You can specify a different path:

```python
from climatextract import extract

result = extract(config_path="./my-custom-config.toml")
```

---

## Input Configuration

Control which PDF files to process:

```toml
[input]
# Option 1: List specific files
filename_list = [
    "data/pdfs/company_2022_report.pdf",
    "data/pdfs/company_2023_report.pdf"
]

# Option 2: Process all PDFs in a directory
# filename_list = "./data/pdfs/sample_reports"
```

---

## Model Configuration

Select which models to use for embedding and extraction:

```toml
[models]
# LLM model for extraction
llm_model = "gpt-4o-mini-2024-07-18"

# Embedding model for semantic search
emb_model = "text-embedding-ada-002"

# Maximum concurrent API calls (adjust based on rate limits)
max_parallel_llm_prompts_running = 4
```

!!! note "Available LLM Models"
    Supported models include: `gpt-4o-mini-2024-07-18`, `gpt-4o-2024-11-20`, `gpt-35-turbo-16k`, `o3-mini-2025-01-31`

---

## Extraction Parameters

Fine-tune the extraction behavior:

```toml
[extraction]
# Year range for emissions data
year_min = 2013
year_max = 2024

# Input mode: "text" or "text+table"
input_mode = "text"

# Prompt type: "default" or "custom_gaia"
prompt_type = "default"

# Semantic search settings
similarity_top_k = 7      # Maximum pages to retrieve
similarity_min_k = 4      # Minimum pages to retrieve
percentile_threshold = 95 # Score cutoff percentile
```

---

## Output Configuration

Control where results are saved:

```toml
[output]
# Base output directory (UUID subdirectory created per run)
output_dir = "output"
```

---

## Evaluation Configuration

Configure evaluation against a gold standard:

```toml
[evaluation]
# Evaluation mode: "no_evaluation", "default", "precision_recall_f1", "both"
evaluation_mode = "no_evaluation"

# Path to gold standard dataset
gold_standard = "data/evaluation_dataset/gist_2025.csv"
```

---

## MLflow Tracking

Optional experiment tracking with MLflow:

```toml
[mlflow]
# Tracking URI: "databricks", "./mlruns", or server URL
tracking_uri = "databricks"

# Experiment name
experiment_name = "/Shared/Experiments/my_experiment"
```

---

## Full Example

Here's a complete configuration file:

```toml
[input]
filename_list = ["data/pdfs/company_2023_report.pdf"]

[models]
llm_model = "gpt-4o-mini-2024-07-18"
emb_model = "text-embedding-ada-002"
max_parallel_llm_prompts_running = 4

[extraction]
year_min = 2018
year_max = 2024
input_mode = "text"
prompt_type = "default"

[output]
output_dir = "output"

[evaluation]
evaluation_mode = "no_evaluation"

[mlflow]
tracking_uri = "./mlruns"
```
