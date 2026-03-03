# Configuration

climatextract uses a TOML configuration file (`climatextract.toml`) to control all aspects of extraction. This guide explains each option.

---

## Configuration File

Create a `climatextract.toml` file in your working directory. You can specify a different path:

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
    "/data/pdfs/company_2022_report.pdf",
    "/data/pdfs/company_2023_report.pdf"
]

# Option 2: Process all PDFs in a directory
# filename_list = "./data/pdfs/"
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
    Supported models include: `gpt-4o-mini-2024-07-18`, `gpt-4o-2024-11-20`, `gpt-35-turbo-16k`, `o3-mini-2025-01-31`, `gpt-4.1-2025-04-14`, `gpt-5-chat-2025-08-07`, `gpt-oss-120b`

If `max_parallel_llm_prompts_running` is not set, the following model-specific defaults apply:

| Model | Default concurrency |
|-------|-------------------|
| `gpt-4o-mini-2024-07-18` | 25 |
| `gpt-4o-2024-11-20` | 25 |
| `gpt-oss-120b` | 25 |
| `gpt-35-turbo-16k` | 8 |
| `gpt-5-chat-2025-08-07` | 8 |
| `gpt-4.1-2025-04-14` | 4 |
| `o3-mini-2025-01-31` | 2 |

---

## Extraction Parameters

Fine-tune the extraction behavior:

```toml
[extraction]
# Year range for emissions data
year_min = 2013
year_max = 2024

# Input mode: "text+table" (default) or "text"
input_mode = "text+table"

# Only embed documents, skip extraction (default: false)
embed_only = false

# Prompt type: "default" or "custom_gaia"
prompt_type = "default"

# Semantic search settings
percentile_threshold = 95 # Score cutoff percentile. Keep 5% most similar pages and discard 95%
similarity_top_k = 7      # Maximum pages to retrieve, overrides percentile_threshold for long documents
similarity_min_k = 4      # Minimum pages to retrieve, overrides percentile_threshold for short documents

# Context window for semantic search (default: 0, meaning that no adjacent pages are used)
context_window = 0

# Custom path to a DuckDB embeddings file (optional)
# If omitted, uses default: data/processed/embeddings/{emb_model}_from_2025_03_06.duckdb
# embeddings_repository = "./data/processed/embeddings/custom_embeddings.duckdb"
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
gold_standard = "/data/evaluation_dataset/gold_standard.csv"
```

---

## Datalake Configuration

Optional Azure Blob Storage paths for shared PDF files and embedding databases:

```toml
[datalake]
# Blob paths: "container_name" or "container_name/subfolder/path"
blob_path_pdfs = "pdfs"
blob_path_embeddings = "embeddings"
```

See [Datalake configuration](./datalake-configuration.md) for details on setting up and using the data lake.

---

## MLflow Tracking

Optional experiment tracking with MLflow:

```toml
[mlflow]
# Tracking URI: "databricks", "./mlruns", or server URL
tracking_uri = "databricks"
TODO: tracking_uri should not be part of the configuration file, but this should live in .env

# Experiment name
experiment_name = "/Shared/Experiments/my_experiment"
```

See [MLflow setup](./mlflow-setup.md) for details and how to set this up.

---

## Full Example

Here's a complete configuration file showing all available options:

```toml
[input]
filename_list = ["/data/pdfs/company_2023_report.pdf"]

[models]
llm_model = "gpt-4o-mini-2024-07-18"
emb_model = "text-embedding-ada-002"
max_parallel_llm_prompts_running = 4  # omit to use model-specific default

[extraction]
year_min = 2018
year_max = 2024
input_mode = "text+table"             # "text+table" (default) or "text"
embed_only = false                    # only embed, skip extraction
prompt_type = "default"               # "default" or "custom_gaia"
context_window = 0                    # context window for semantic search
similarity_top_k = 7                  # max pages to retrieve
similarity_min_k = 4                  # min pages to retrieve
percentile_threshold = 95             # score cutoff percentile
# embeddings_repository = "./data/processed/embeddings/custom_embeddings.duckdb"

[output]
output_dir = "output"

[evaluation]
evaluation_mode = "no_evaluation"     # "no_evaluation", "default", "precision_recall_f1", "both"
gold_standard = "data/evaluation_dataset/gold_standard.csv"

[datalake]
blob_path_pdfs = "pdfs"              # "container" or "container/subfolder"
blob_path_embeddings = "embeddings"  # "container" or "container/subfolder"

[mlflow]
tracking_uri = "./mlruns"             # "databricks", "./mlruns", or server URL
experiment_name = "climatextract_experiments"
```
