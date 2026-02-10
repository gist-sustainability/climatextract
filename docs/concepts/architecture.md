# Architecture

climatextract uses a Retrieval-Augmented Generation (RAG) architecture to extract emissions data from PDF reports. This page provides a high-level overview of how the components work together.

---

## Pipeline Overview

```mermaid
flowchart LR
    A[PDF Reports] --> B[Text Extraction]
    B --> C[Embedding]
    C --> D[(Vector Database)]
    E[Search Query] --> F[Semantic Search]
    D --> F
    F --> G[Relevant Pages]
    G --> H[LLM Prompt]
    H --> I[GPT Model]
    I --> J[Structured Output]
    J --> K[Post-Processing]
    K --> L[CSV Results]
```

---

## Core Components

### 1. PDF Processing

The pipeline starts by extracting text and tables from PDF pages:

- Uses Docling for text and table extraction
- Each page is processed independently
- Maintains page number metadata for traceability

### 2. Embedding & Storage

Text is converted to vector embeddings for semantic search:

- Default model: `text-embedding-ada-002` (OpenAI)
- Embeddings stored in DuckDB for efficient retrieval
- Query embeddings are cached to avoid redundant API calls

### 3. Semantic Search

When processing a document, the pipeline:

1. Embeds the search query (e.g., "What are the total CO2 emissions in different years? Include Scope 1, Scope 2, and Scope 3 emissions if available.")
2. Computes cosine similarity against all page embeddings
3. Retrieves top-k most relevant pages

### 4. LLM Extraction

Relevant pages are passed to a large language model:

- Structured prompts define the extraction task
- Output parsed via Pydantic models (`custom_gaia`) or regex (`default`)
- Each scope-year combination is extracted independently

### 5. Post-Processing

Raw LLM output is cleaned and structured:

- Unit normalization (e.g., "tonnes" → "tCO2e")
- Duplicate resolution across pages
- Value standardization and validation

---

## Key Classes

| Class | Purpose |
|-------|---------|
| `ValueRetrieverPipeline` | Orchestrates the full extraction workflow |
| `EmbeddingsRepository` | Manages DuckDB storage for embeddings |
| `CustomPromptGaia` | Structures LLM prompts with Pydantic parsing |
| `EvaluatorPrecisionRecallF1` | Computes evaluation metrics |
| `Pdfdoc` | PDF document representation with page data |
| `Llm` | LLM client with token counting and rate limiting |
| `DataLakeManager` | Manages PDF downloads and file checks |

---

## Data Flow

```mermaid
sequenceDiagram
    participant User
    participant Extract API
    participant Pipeline
    participant Vector DB
    participant LLM

    User->>Extract API: extract(pdf_path)
    Extract API->>Pipeline: Process PDF
    Pipeline->>Vector DB: Store/retrieve embeddings
    Pipeline->>Vector DB: Semantic search
    Vector DB-->>Pipeline: Relevant pages
    Pipeline->>LLM: Extract emissions
    LLM-->>Pipeline: Structured data
    Pipeline->>Extract API: Save results
    Extract API-->>User: Result path
```

---

## Next Steps

- [RAG Pipeline](rag-pipeline.md) – Deep dive into retrieval and generation
- [Prompts](prompts.md) – How prompts are structured
- [Evaluation](evaluation.md) – Measuring extraction quality
