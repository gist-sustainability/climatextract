# Understanding Output

After running extraction, climatextract saves results to the `output/<run-id>/` directory. This guide explains the output files and their contents.

---

## Output Directory Structure

```
output/
└── abc123-uuid/
    ├── results_long.csv       # Main results (long format)
    ├── results_wide.csv       # Results pivoted by scope
    ├── query_responses.csv    # Page-level details
    └── evaluation/            # (if evaluation enabled)
        ├── comparison.csv
        └── metrics.csv
```

---

## Main Results: `results_long.csv`

The primary output file with one row per extracted value:

| Column | Description |
|--------|-------------|
| `short_report_name` | Filename of the PDF |
| `year` | Year of the emissions data |
| `scope` | Scope type (1, 2, or 3) |
| `value` | Extracted emissions value |
| `unit` | Unit of measurement (e.g., `tCO2e`) |
| `page_number` | Page where value was found |
| `confidence` | LLM confidence score (if available) |

**Example:**

| short_report_name | year | scope | value | unit | page_number |
|-------------------|------|-------|-------|------|-------------|
| apple_2021_en.pdf | 2021 | 1 | 55000 | tCO2e | 42 |
| apple_2021_en.pdf | 2021 | 2 | 120000 | tCO2e | 42 |
| apple_2021_en.pdf | 2020 | 1 | 52000 | tCO2e | 43 |

---

## Wide Format: `results_wide.csv`

The same data pivoted for easier comparison across scopes:

| Column | Description |
|--------|-------------|
| `short_report_name` | Filename of the PDF |
| `year` | Year of the emissions data |
| `scope_1` | Scope 1 emissions value |
| `scope_2` | Scope 2 emissions value |
| `scope_3` | Scope 3 emissions value |

---

## Query Responses: `query_responses.csv`

Detailed page-level information about the extraction process:

| Column | Description |
|--------|-------------|
| `short_report_name` | Filename of the PDF |
| `page_number` | Page number analyzed |
| `similarity_score` | Semantic similarity to search query |
| `llm_response` | Raw LLM output |

---

## Evaluation Output

When evaluation is enabled, additional files are created:

### `comparison.csv`

Row-by-row comparison with gold standard:

| Column | Description |
|--------|-------------|
| `match_type` | `true_positive`, `false_positive`, `false_negative` |
| `extracted_value` | Value from pipeline |
| `expected_value` | Value from gold standard |

### `metrics.csv`

Aggregate evaluation metrics:

| Metric | Description |
|--------|-------------|
| `precision` | Correct extractions / total extractions |
| `recall` | Correct extractions / total expected |
| `f1_score` | Harmonic mean of precision and recall |

---

## Handling Duplicates

The pipeline may extract the same value from multiple pages. Duplicates are handled by:

1. **Keeping the highest confidence** value when available
2. **Preferring earlier pages** when confidence is equal
3. **Flagging duplicates** with a `duplicate_reason` column

!!! tip "Duplicate Investigation"
    Use `query_responses.csv` to investigate why duplicates occurred and which pages contained the data.
