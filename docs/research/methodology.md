# Methodology

This page describes the scientific methodology behind climatextract, suitable for academic citations and reproducibility.

---

## Task Definition

**Objective:** Given a corporate sustainability report (PDF), extract all reported CO₂ emissions values for Scope 1, 2, and 3 across available years.

**Output:** Structured table with columns: `year`, `scope`, `value`, `unit`

---

## Approach: Retrieval-Augmented Generation

We use a RAG pipeline consisting of:

### 1. Document Preprocessing

- PDF pages are extracted as text using `pdf2image` with OCR fallback
- Each page is treated as an independent document chunk
- Metadata (page number, filename) is preserved for traceability

### 2. Semantic Retrieval

- Pages are embedded using OpenAI's `text-embedding-ada-002` model
- A task-specific query ("CO₂ emissions Scope 1 2 3...") is embedded
- Cosine similarity identifies the most relevant pages
- Top-k pages are selected based on score thresholds

### 3. LLM Extraction

- Selected pages are passed to GPT models with structured prompts
- Prompts include:
  - Role definition (climate analyst)
  - KPI definitions (Scope 1, 2, 3)
  - Extraction constraints (company-level, absolute values only)
- Output is constrained to JSON format with Pydantic validation

### 4. Post-Processing

- Unit normalization (e.g., "tonnes CO2" → "tCO2e")
- Duplicate resolution across pages
- Value validation and standardization

---

## Evaluation Methodology

### Gold Standard

- Human-annotated dataset of emissions values
- Covers diverse companies and reporting formats
- Annotations include page-level source references

### Metrics

We report standard information extraction metrics:

- **Precision:** Fraction of extracted values that are correct
- **Recall:** Fraction of gold standard values that were extracted
- **F1 Score:** Harmonic mean of precision and recall

### Matching Criteria

An extraction is considered correct if:

1. Year matches exactly
2. Scope matches exactly
3. Value matches (with tolerance for minor differences)
4. Unit is semantically equivalent after normalization

---

## Reproducibility

All experiments are logged with:

- **MLflow:** Parameters, metrics, and artifacts
- **Configuration files:** `climxtract.toml` snapshots
- **Version control:** Git commit hashes

To reproduce results:

1. Use the same `climxtract.toml` configuration
2. Ensure identical model versions (specified in config)
3. Run against the same gold standard dataset

---

## Limitations

- **Language:** Currently optimized for English reports
- **Format:** Complex tables may not be fully captured
- **Coverage:** Some disclosure formats may be missed
- **Model dependency:** Results may vary with different LLM versions

---

## Future Work

- Multi-language support
- Improved table extraction
- Fine-tuned extraction models
- Expanded KPI coverage beyond emissions
