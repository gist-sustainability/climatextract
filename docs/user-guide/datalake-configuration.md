# Sharing Large Files

Use Azure Blob Storage to share PDF files and embedding databases across your team. This is optional — if all your files are available locally, you don't need a data lake.

---

## Prerequisites

- An **Azure Blob Storage** account
- Either **Azure credentials** configured (same authentication used for LLM access — see [Installation](../getting-started/installation.md)) or a **SAS token** handed to you by whoever administers the storage account

---

## Setup

Add your storage account URL to the `.env` file:

```bash
AZURE_STORAGE_ACCOUNT_URL=https://<your-storage-account>.blob.core.windows.net/

# optional
AZURE_STORAGE_SAS_TOKEN=<your-Shared-Access-Signatore-token>
```

The `AZURE_STORAGE_SAS_TOKEN` for authentication is optional. It is only needed if you have no Entra ID access and were given the token instead. It is a private key, so keep it secret. It expires after x days/months.

That's it. climatextract will automatically offer to download missing files from the data lake when you run an extraction.

---

## Blob Storage Structure

By default, climatextract expects two containers in your storage account, each with a **flat structure** (no subdirectories):

```
<your-storage-account>
├── pdfs/
│   ├── company_a_2022_report.pdf
│   ├── company_b_2023_report.pdf
│   └── ...
└── embeddings/
    ├── <embedding-model>_from_<date>.duckdb
    └── ...
```

| Container | Contents | Naming convention |
|-----------|----------|-------------------|
| `pdfs` | PDF sustainability reports | Filename only, e.g. `company_2022_report.pdf` |
| `embeddings` | Pre-computed DuckDB vector databases | `{embedding_model}_from_{date}.duckdb`, e.g. `text-embedding-ada-002_from_2025_12_23.duckdb` |

!!! important
    Files are looked up by **filename only**. If your local path is `data/pdfs/company_2022_report.pdf`, climatextract looks for a blob named `company_2022_report.pdf` within the configured blob path. The local directory structure does not matter.

---

## Custom Blob Paths

If your files are stored in different containers or inside subfolders, configure the blob paths in `climatextract.toml`:

```toml
[datalake]
# Default: files at the root of "pdfs" and "embeddings" containers
blob_path_pdfs = "pdfs"
blob_path_embeddings = "embeddings"

# Example: files inside subfolders
# blob_path_pdfs = "mycontainer/sustainability/reports"
# blob_path_embeddings = "mycontainer/models/embeddings"
```

The format is `container_name` or `container_name/subfolder/path`. For example, setting `blob_path_pdfs = "data/reports/2023"` means climatextract will look for PDFs in the `data` container under the `reports/2023/` prefix.

---

## How Downloads Work

When you run an extraction, climatextract checks what's available locally and prompts you to download anything that's missing:

1. **Embedding database** — If the DuckDB file doesn't exist locally, you're asked whether to download it from the embeddings blob path. This is checked first, since an existing database may already contain the pages you need.

2. **PDF files** — Which PDFs need downloading depends on the input mode:

    | Input mode | What gets downloaded |
    |------------|---------------------|
    | `text` | Only PDFs that aren't already in the embedding database. If a PDF's pages are already embedded, the file itself isn't needed. |
    | `text+table` | All PDFs that aren't on disk, regardless of embedding status. Table extraction requires the original PDF file. |

You are always prompted before any download begins. The prompt shows the number of files and total download size.

---

## Uploading Files (Superuser)

climatextract only *downloads* from the data lake. To upload files, use the Azure CLI, Azure Portal, or Azure Storage Explorer.

**Using Azure CLI:**

```bash
# Upload a single PDF
az storage blob upload \
    --account-name <your-storage-account> \
    --container-name pdfs \
    --file ./data/pdfs/company_2022_report.pdf \
    --name company_2022_report.pdf \
    --auth-mode login

# Upload all PDFs in a directory
az storage blob upload-batch \
    --account-name <your-storage-account> \
    --destination pdfs \
    --source ./data/pdfs/ \
    --auth-mode login

# Upload an embedding database
az storage blob upload \
    --account-name <your-storage-account> \
    --container-name embeddings \
    --file ./data/processed/embeddings/text-embedding-ada-002_from_2025_12_23.duckdb \
    --name text-embedding-ada-002_from_2025_12_23.duckdb \
    --auth-mode login
```

!!! tip
    When sharing a pre-built embedding database, team members can skip the embedding step entirely — saving time and API costs.

---

## Working Without a Data Lake

If `AZURE_STORAGE_ACCOUNT_URL` is not set in your `.env` file, the data lake is silently skipped. Everything runs locally:

- PDFs must already exist on disk at the paths specified in your configuration
- The embedding database is created from scratch if it doesn't exist

This is the default behavior and requires no additional setup.

---

## Next Steps

- [Architecture](../concepts/architecture.md) – Understand how the pipeline works
- [RAG Pipeline](../concepts/rag-pipeline.md) – Deep dive into retrieval and generation
