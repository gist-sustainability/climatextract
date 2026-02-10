# Understanding Output

After running extraction, climatextract saves results to the `output/<run-id>/` directory. This guide explains the output files and their contents.

---

## Output Directory Structure

```
output/
└── abc123-uuid/
    ├── 03_co2_emission_table2_w_query_responses.csv           # Page-level details (with duplicates)
    ├── 03_co2_emission_table2_w_query_responses_filtered.csv  # Deduplicated responses
    ├── intermediate_results.csv                               # Intermediate extraction results
    ├── results_long_format.csv                                # Main results (long format)
    ├── results_wide_format.csv                                # Results pivoted by year
    ├── invalid_llm_outputs.txt                                # Invalid LLM responses
    ├── logs.json                                              # Parameters, metrics, run info
    ├── 04a_results_available_in_report.csv                    # (if evaluation enabled)
    ├── 04b_results_not_available_in_report.csv                # (if evaluation enabled)
    └── 05_results_aggregated_by_*.csv                         # (if evaluation enabled)
```

---

## Main Results: `results_long_format.csv`

The primary output file with one row per extracted value:

| Column | Description |
|--------|-------------|
| `report_id` | Filename of the PDF |
| `year` | Year of the emissions data |
| `indicator` | Scope type (`scope 1`, `scope 2lb`, `scope 2mb`, `scope 3`) |
| `value_std` | Standardized emissions value |
| `unit_std` | Standardized unit (always `t CO2e` for scope indicators) |
| `page` | Page where value was found |

Additional detail columns: `value_raw` (raw extracted value), `value_score` (LLM confidence), `unit_raw` (raw extracted unit), `unit_score` (unit confidence), `unit_cat` (unit category), `dupl_flag` (duplicate flag), `select_flag` (selection flag).

**Example:**

| report_id | year | indicator | value_std | unit_std | page |
|-----------|------|-----------|-----------|----------|------|
| sato holdings_2022_report.pdf | 2015 | scope 1 | 135.0 | t CO2e | 34 |
| sato holdings_2022_report.pdf | 2015 | scope 2lb | 41962.0 | t CO2e | 34 |
| sato holdings_2022_report.pdf | 2015 | scope 2mb | 37674.0 | t CO2e | 34 |

---

## Wide Format: `results_wide_format.csv`

The same data pivoted for easier comparison across scopes. Key columns:

| Column | Description |
|--------|-------------|
| `report_id` | Filename of the PDF |
| `year` | Year of the emissions data |
| `scope_1_value_std` | Scope 1 standardized emissions value |
| `scope_2lb_value_std` | Scope 2 (location-based) standardized emissions value |
| `scope_2mb_value_std` | Scope 2 (market-based) standardized emissions value |
| `scope_3_value_std` | Scope 3 standardized emissions value |

Each scope also has additional detail columns following the pattern `scope_{N}_{field}`, where `N` is `1`, `2lb`, `2mb`, or `3`, and `{field}` is one of: `value_raw`, `value_score`, `unit_std`, `unit_raw`, `unit_score`, `unit_cat`, `dupl_reason`, `page`.

---

## Query Responses: `03_co2_emission_table2_w_query_responses.csv`

Detailed page-level information about the extraction process. Key columns:

| Column | Description |
|--------|-------------|
| `report_name_short` | Filename of the PDF |
| `page_number_used_by_llm` | Page number analyzed |
| `page_retrieval_scores` | Semantic similarity scores for page retrieval |
| `extracted_value_from_llm` | Value extracted by the LLM |
| `extracted_scope_from_llm` | Scope extracted by the LLM |
| `extracted_year_from_llm` | Year extracted by the LLM |
| `extracted_unit_from_llm` | Unit extracted by the LLM |
| `raw_llm_response` | Raw LLM output |
| `value_probability` | LLM confidence for the value |
| `unit_probability` | LLM confidence for the unit |
| `normalized_unit_from_dictionary` | Unit after dictionary normalization |
| `standardized_value` | Value after standardization |
| `duplicate_flag` | Whether the row is a duplicate |
| `select_flag` | Whether the row was selected after deduplication |
| `dupl_reason` | Reason for duplicate resolution decision |

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

## Handling Duplicates

The pipeline may extract the same value from multiple pages. Duplicates are resolved using three prioritization rules applied in order:

1. **Identical entries**: Drop duplicate rows with the same value and unit on the same page
2. **Preferred unit**: Keep entries with the preferred unit (`t CO2e`) when available
3. **Majority page**: Keep entries from the page with the most matches

The deduplicated results are saved in `03_co2_emission_table2_w_query_responses_filtered.csv`.

!!! tip "Duplicate Investigation"
    Use `03_co2_emission_table2_w_query_responses.csv` to investigate why duplicates occurred and which pages contained the data.
