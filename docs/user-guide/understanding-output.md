# Understanding Output

After running extraction, climatextract saves results to the `output/<run-id>/` directory. This guide explains the output files and their contents.

---

## Output Directory Structure

```
output/
└── abc123-uuid/
    ├── raw_results.csv           # Page-level details (with duplicates)
    ├── raw_results_temp.csv                               # Intermediate extraction results
    ├── results_long_format.csv                                # Main results (long format, with duplicates)
    ├── results_wide_format.csv                                # Results pivoted by year
    ├── config.json (or config_and_metrics.json)                                             # Parameters, metrics, run info
    ├── 04a_results_available_in_report.csv                    # (if evaluation enabled)
    ├── 04b_results_not_available_in_report.csv                # (if evaluation enabled)
    └── 05_results_aggregated_by_*.csv                         # (if evaluation enabled)
```

---

## Main Results: `results_long_format.csv`

The primary output file with one row per extracted value. No row included if the LLM did not extract the desired value (=rows are dropped if `value_raw` is NA) :

| Column | Description |
|--------|-------------|
| `report_id` | Filename of the PDF |
| `year` | Year of the emissions data |
| `indicator` | Scope type (`scope 1`, `scope 2lb`, `scope 2mb`, `scope 3`) |
| `value_std` | Standardized emissions value |
| `unit_std` | Standardized unit (always `t CO2e` for scope indicators) |
| `page` | Page counter where value was found (may differ from printed page numbers) |
| `dupl_flag` | Duplicate flag, indicates presence of duplicates, i.e.,  multiple, possibly conflicting values for a given report_id-year-indicator combination, see details below |
| `select_flag` | Selection flag from the duplicate resolution mechanism, indicates the selected row containing the preferred value, see details below |
| | **Further details from extraction process** |
| `value_raw` | Original extracted value from LLM
| `value_score` | LLM confidence score, based on logprobs
| `unit_raw` | Original extracted unit from LLM
| `unit_score` |  LLM confidence score, based on logprobs
| `unit_cat` | Categorized unit (e.g., `kg CO2e`, `Mt CO2e`, `Other`, `Unknown`). The [unit conversion table](https://github.com/gist-sustainability/climatextract/blob/main/data/normalization_units/unit_normalization_dict.csv) converts `unit_raw` to `unit_cat`, so that standardized values (in `t CO2e`) can be calculated.


**Example:**

| report_id | year | indicator | value_std | ... | unit_std | ... | page |
|-----------|------|-----------|-----------|-----|----------|-----|------|
| company_2023_report.pdf | 2015 | scope 1 | 135.0 | ... | t CO2e | ... | 34 |
| company_2023_report.pdf | 2015 | scope 2lb | 41962.0 | ... | t CO2e | ... | 34 |
| company_2023_report.pdf | 2015 | scope 2mb | 37674.0 | ... | t CO2e | ... | 34 |
| company_2023_report.pdf | 2015 | scope 3 | 1834.0 | ... | t CO2e | ... | 34 |
| company_2023_report.pdf | 2016 | scope 1 | 170.0 | ... | t CO2e | ... | 34 |

---

## Wide Format: `results_wide_format.csv`

The same data pivoted for easier comparison across scopes: Wide format, with a single row per report_id and year. Key columns:

| Column | Description |
|--------|-------------|
| `report_id` | Filename of the PDF |
| `year` | Year of the emissions data |
| `scope_1_value_std` | Scope 1 standardized emissions value |
| `scope_2lb_value_std` | Scope 2 (location-based) standardized emissions value |
| `scope_2mb_value_std` | Scope 2 (market-based) standardized emissions value |
| `scope_3_value_std` | Scope 3 standardized emissions value |

Each scope also has additional detail columns following the pattern `scope_{type}_{field}`, where `type` is `1`, `2lb`, `2mb`, or `3`, and `{field}` is one of: `value_raw`, `value_score`, `unit_std`, `unit_raw`, `unit_score`, `unit_cat`, `dupl_reason`, `page`.

`dupl_reason` (e.g., `scope_1_dupl_reason`) contains the priorization rule used during duplicate resolution. It is related to `dupl_flag` and `select_flag` in the long format. The wide format contains only the selected/resolved values from the long format, with `dupl_reason` explaining how each duplicate was resolved.

---

#### Handling Duplicates

For a given indicator, the pipeline may extract multiple identical/conflicting values from multiple pages of a PDF report. These duplicates are resolved using three prioritization rules applied in order:

1. **Identical entries**: Drop duplicate rows with the same value and unit on the same page, sets `dupl_reason = 1`
2. **Preferred unit**: Keep entries with the preferred unit (`t CO2e`) when available, sets `dupl_reason = 2`
3. **Majority page**: Keep entries from the page with the most matches, sets `dupl_reason = 3`

In cases when only a single value per report_id-year combination got extracted, no duplicate resolution is necessary: `dupl_reason = 0`

!!! tip "Duplicate Investigation"
    Use `results_wide_format.csv` to investigate why duplicates occurred and which pages contained the data.

---

## Query Responses: `raw_results.csv`

Detailed page-level information about the extraction process. 

Expect one row for each indicator-year-page combination. For example, if 5 pages from a single document are being analyzed and you want the LLM to extract 4 indicators from the last 10 years, one would expect 5 * 4 * 10 = 200 rows. In many rows the desired value will be `Not specified`, as mentioned by the LLM. Row numbers can (slightly) deviate from this calculation if the LLM extracts two or more values from a given page or returns no information about the value, not even `Not specified`.

Key columns (column names are not final):

| Column | Same as | Description |
|--------|---------| -------------|
| `report_name_short` | `report_id` | Filename of the PDF |
| `page_number_used_by_llm` | `page` | Page number analyzed |
| `page_retrieval_scores` | | Semantic similarity scores for page retrieval |
| `page_texts_to_llm` | | Text from page given to LLM |
| `raw_llm_response` | | Raw LLM answer output |
| `extracted_scope_from_llm` | $\hat{=}$ `indicator` | Scope extracted by the LLM |
| `extracted_year_from_llm` | `year` | Year extracted by the LLM |
| `extracted_value_from_llm_orig` | | Value extracted by the LLM |
| `extracted_value_from_llm` | `value_raw` | Value extracted by the LLM after minimal processing |
| `standardized_value` | `value_std` | Value after standardization, measured in `t CO2e` |
| `extracted_unit_from_llm` | `unit_raw` | Unit extracted by the LLM |
| `normalized_unit_from_dictionary` | `unit_cat` | Unit after dictionary normalization |
| `value_probability` | `value_score` | LLM confidence for the extracted value |
| `unit_probability` | `unit_score` | LLM confidence for the extracted unit |
| `duplicate_flag` | `dupl_flag` | Whether the row is a duplicate |
| `select_flag` | `select_flag` | Whether the row was selected after deduplication |

---

## Logs

### `config.json` or `config_and_metrics.json`

Stores the configuration, incured costs, and evaluation metrics if evaluation is configured. A summary of what happend during this run. 
When evaluation is enabled, the file is called `config.json`, otherwise `config_and_metrics.json`. 

If [MLflow](./mlflow-setup.md) has been activated, the same information will also be saved in Mlflow for comparison between experiments.

---

## Evaluation Output

When evaluation is enabled, additional files are created directly in the run directory:

### `04a_results_available_in_report.csv`

Values where emissions information exists in the report — row-by-row comparison with gold standard.

### `04b_results_not_available_in_report.csv`

Values where emissions information does not exist in the report.

### `05_results_aggregated_by_*.csv`

Aggregate evaluation metrics grouped by different dimensions (e.g., per document, per scope).

### Precision-Recall-F1 Mode

When using `precision_recall_f1` or `both` evaluation mode, additional files are created:

- `error_analysis_per_doc.csv` — Error analysis aggregated per document
- `error_analysis_per_row.csv` — Error analysis per individual row

---

## Next Steps

- [MLflow Setup](mlflow-setup.md) – Configure experiment tracking
- [Sharing Large Files](datalake-configuration.md) – Share PDFs and embeddings with your team
