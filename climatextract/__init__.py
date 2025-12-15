"""
ClimXtract - Extract CO2 emissions data from PDF sustainability reports.

Public API:
    extract(pdf_input) - Extract emissions data, returns path to results
    extract_and_evaluate(pdf_input, gold_standard_path) - Extract and evaluate against gold standard

Configuration:
    Create a `climxtract.toml` file in your project root to configure the extraction.
    See the package documentation for available options.

Example:
    from climatextract import extract, extract_and_evaluate
    
    # Simple extraction
    results_path = extract("./reports/")
    
    # Extraction with evaluation
    results_path = extract_and_evaluate(
        "./reports/",
        gold_standard_path="./gold_standard.csv"
    )
"""

from climatextract.main import extract, extract_and_evaluate

__version__ = "0.1.0"
__all__ = ["extract", "extract_and_evaluate"]

