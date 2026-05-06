"""
ClimXtract - Extract CO2 emissions data from PDF sustainability reports.

Public API:
    extract(pdf_input, enable_mlflow, config_path, verbose) - Extract emissions data
    extract_and_evaluate(pdf_input, gold_standard_path, enable_mlflow, config_path, verbose) - Extract and evaluate

Configuration:
    Create a `climatextract.toml` file in your project root to configure the extraction.
    See the package documentation for available options.

Example:
    from climatextract import extract, extract_and_evaluate

    # Simple extraction (no MLflow)
    results_path = extract("./reports/")

    # Extraction with full MLflow tracking (params, metrics, artifacts, traces, OpenAI calls)
    results_path = extract("./reports/", enable_mlflow=True)

    # Extraction with evaluation
    results_path = extract_and_evaluate(
        "./reports/",
        gold_standard_path="./gold_standard.csv"
    )

    # Extraction with evaluation and MLflow tracking
    results_path = extract_and_evaluate(
        "./reports/",
        gold_standard_path="./gold_standard.csv",
        enable_mlflow=True
    )

    # Verbose mode with detailed per-PDF output
    results_path = extract("./reports/", verbose=True)
"""

import warnings
import os

# Disable MLflow artifact upload progress bar
os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"

# Suppress transformers/tokenizers output before they are imported
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress all warnings from these modules
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="urllib3")
warnings.filterwarnings("ignore", message=".*Creating a trace within the default experiment.*")
warnings.filterwarnings("ignore", message=".*mlflow.tracing.*")
# llama-index's ``Field(validate_default=True)`` usage is flagged by
# pydantic 2.12+. Harmless, will self-resolve when llama-index updates.
warnings.filterwarnings("ignore", message=".*validate_default.*")

# Now import pandas and set option
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

from climatextract.main import extract, extract_and_evaluate

__version__ = "0.3.0"
__all__ = ["extract", "extract_and_evaluate"]

