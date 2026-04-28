# Quickstart

Get up and running with climatextract in just a few minutes.

---

## Basic Extraction

The simplest way to extract emissions data from a PDF:

```python
from climatextract import extract

# Extract from a single PDF
result_path = extract("./data/pdfs/company_2023_report.pdf")
print(f"Results saved to: {result_path}")
```

This will:

1. Embed the PDF pages into a vector database
2. Search for pages relevant to CO₂ emissions
3. Use an LLM to extract Scope 1, 2, and 3 data
4. Save results to the `output/` directory

Many corporate annual reports / sustainability reports are available for download online, for example at the [https://www.srnav.com/reports](https://www.srnav.com/reports) repository.

---

## Extract from Multiple PDFs

Process an entire directory of reports:

```python
from climatextract import extract

# Process all PDFs in a directory
result_path = extract("./data/pdfs/")
```

Or provide a specific list:

```python
from climatextract import extract

files = [
    "./data/pdfs/report1.pdf",
    "./data/pdfs/report2.pdf",
]
result_path = extract(files)
```

---

## Extract and Evaluate

If you have a gold standard dataset, validate your results:

```python
from climatextract import extract_and_evaluate

result_path = extract_and_evaluate(
    pdf_input="./data/pdfs/",
    gold_standard_path="./data/evaluation_dataset/gold_standard.csv"
)
```

Or check out our gold-standard dataset: [Paper](https://doi.org/10.1038/s41597-025-05664-8), [Data](https://zenodo.org/records/18696096)

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

## Working directory structure

In our projects, the working directory is typically organized as follows. This structure is used throughout this documentation.

```
<your-working-directory>/
├── data/
    ├── evaluation_dataset/             # (optional) Benchmark datasets for evaluation.
    ├── pdfs/                           # Directory with PDF files (annual reports / sustainability reports). Move your files here.
    ├── processed/                      # (automatically generated) PDF files after initial processing
        ├── embeddings/                 # (automatically generated) Directory with .duckdb databases. Stores embeddings and raw text from each page in PDF files
        └── tables/                     # (automatically generated, only used if input_mode = "text+table") Stores tables from PDF files, converted with Docling to .csv files
├── output/                             # (automatically generated) Every time when you run extract(), a new subfolder with results is created here
    ├── abc123-uuid/                                       
    └── def456-uuid/
├── climatextract.toml                  # ClimXtract Configuration File to override defaults
└── .env                                # Contains secret values about your environment, specific to your local configuration. Never, ever share this file!
```

We recommend that you move your sustainability reports to the ``.data/pdfs/`` folder (or change the default names in the configuration file). 

---

### Teamwork: Populate data folders automatically

If you work in a team, it can be tedious to ensure everone has access to the same PDF files and embedding databases. Our solution:

- One superuser stores all the PDF files (and embedding databases) online in an Azure datalake.
- Everyone who has access can download the files from the datalake on demand as needed. No need to transfer, copy and paste PDF files by hand, and everyone uses the same database.

Add to your `.env` file:

```bash
AZURE_STORAGE_ACCOUNT_URL=https://<your-datalake-name>.blob.core.windows.net/
```

See [Sharing Large Files](../user-guide/datalake-configuration.md) for details.

---

## Output Structure

After extraction, you'll find results in `output/<run-id>/`:

```
output/
└── abc123-uuid/
    ├── raw_results.csv             # Page-level LLM responses (with duplicates)
    ├── raw_results_temp.csv        # Pre-normalization extraction results
    ├── results_long_format.csv     # Main results in long format (with duplicates)
    ├── results_wide_format.csv     # Results pivoted by year
    └── config.json                 # Parameters, metrics, run info
                                    # (config_and_metrics.json when using extract_and_evaluate)
```

See [Understanding Output](../user-guide/understanding-output.md) for column definitions.

---

## Next Steps

- [Configuration](../user-guide/configuration.md) – Customize extraction behavior
- [Architecture](../concepts/architecture.md) – Understand how the pipeline works
