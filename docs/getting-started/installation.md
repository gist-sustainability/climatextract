# Installation

This guide walks you through setting up climatextract on your machine.

---

## Prerequisites

Before installing, ensure you have:

- **Python 3.11+** – [Download Python](https://www.python.org/downloads/)
- **pip** – Included with Python 3.4+
- **Azure credentials** – Access to Azure OpenAI services

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/gist-sustainability/climatextract.git
cd climatextract
```

---

## Step 2: Create a Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 4: Install System Dependencies

climatextract uses `pdf2image` which requires **Poppler**:

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

## Step 5: Configure Azure Authentication

Create a `.env` file in the project root with your Azure credentials:

```bash
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
```

!!! note "Authentication Options"
    See the [Azure Auth Helper documentation](https://github.com/soda-lmu/azure-auth-helper-python/blob/main/AuthenticationWorkflowSetup.md) for alternative authentication workflows.

---

## Verify Installation

Test that everything is working:

```bash
python -c "from climatextract import extract; print('Installation successful!')"
```

---

## Next Steps

Ready to extract some data? Head to the [Quickstart](quickstart.md) guide.
