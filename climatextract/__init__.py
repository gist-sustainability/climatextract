"""
ClimXtract - Extract CO2 emissions data from PDF sustainability reports.

Public API:
    extract(pdf_input, enable_mlflow, config_path) - Extract emissions data
    extract_and_evaluate(pdf_input, gold_standard_path, enable_mlflow, config_path) - Extract and evaluate

Configuration:
    Create a `climxtract.toml` file in your project root to configure the extraction.
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
"""

from climatextract.main import extract, extract_and_evaluate

__version__ = "0.2.1"
__all__ = ["extract", "extract_and_evaluate"]

