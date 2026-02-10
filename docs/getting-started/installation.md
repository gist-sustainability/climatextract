# Installation

This guide walks you through installing climatextract.

---

## Prerequisites

Before installing, ensure you have:

- **Python 3.11+** – [Download Python](https://www.python.org/downloads/)
- **Azure credentials** – Access to Azure OpenAI services

---

## Step 1: Install the Package

```bash
pip install climatextract
```

This installs climatextract and all required dependencies.

---

## Step 2: Install System Dependencies

climatextract uses Docling for PDF processing, which requires **Poppler**:

=== "macOS"

    ```bash
    brew install poppler
    ```

=== "Ubuntu/Debian"

    ```bash
    sudo apt-get install poppler-utils
    ```

=== "Windows"

    Download from [poppler releases](https://github.com/osber/poppler-windows/releases) and add to PATH.

---

## Step 3: Configure Azure Authentication

Create a `.env` file in your working directory with your Azure credentials:

```bash
AZURE_ENDPOINT=https://your-endpoint.openai.azure.com/
API_KEY=your-api-key
API_VERSION=2024-12-01-preview
```

---

## Verify Installation

Test that everything is working:

```bash
python -c "from climatextract import extract; print('Installation successful!')"
```

---

## Next Steps

Ready to extract some data? Head to the [Quickstart](quickstart.md) guide.
