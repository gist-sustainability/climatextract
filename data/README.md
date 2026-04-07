# Data Directory

This is the default location where climatextract stores and looks for data artifacts.

## Default Subdirectories

### `processed/embeddings/`
The semantic search module stores DuckDB databases here, named by embedding model and date (e.g., `text-embedding-ada-002_from_2025_03_06.duckdb`). Created automatically when you embed PDFs for the first time.

### `processed/tables/`
Extracted table cells are cached here as CSVs to avoid re-extracting tables on subsequent runs. Created automatically during extraction.

## Recommended Locations

### `pdfs/`
We recommend placing your PDF sustainability reports here. However, PDFs can be stored anywhere — just specify the path via `filename_list` in `climatextract.toml` or as an argument to `extract()`.

### `evaluation_dataset/`
We recommend placing your gold standard CSV here if you want to use `extract_and_evaluate()`. However, the gold standard file can be stored anywhere — just specify the path via `gold_standard` in `climatextract.toml` or as an argument to `extract_and_evaluate()`.
