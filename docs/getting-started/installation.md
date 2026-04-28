# Installation

This guide walks you through installing climatextract.

---

## Prerequisites

Before installing, ensure you have:

- **Python 3.11+** – [Download Python](https://www.python.org/downloads/)
- **Azure credentials** – Access to Azure AI Foundry, if using the default adapter (see step 3). Not required if you're injecting your own provider handler — see [Custom Providers](../user-guide/custom-providers.md).

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

## Step 3: Configure Access to Large Language Models via Azure

You will need to set up a Large Language Model in Azure. This package supports some models via Azure OpenAI and others via Azure's AI foundry.

In addition, set up an embedding model that is accessible via ``AZURE_ENDPOINT`` and named ``text-embedding-ada-002``.<!-- TODO(litellm-refactor): local HuggingFace sentence-transformers support was removed when config.py was retired. Restore this sentence once a HuggingFace EmbeddingModelHandler adapter lands.
 Alternatively, you can use a local HuggingFace ``sentence-transformers/*`` model, which does not require Azure (see [Configuration](../user-guide/configuration.md)).
-->

The default handler is Azure AI Foundry — see [`climatextract/adapters/azure_ai_foundry.py`](https://github.com/gist-sustainability/climatextract/blob/main/climatextract/adapters/azure_ai_foundry.py). The package also ships an Azure OpenAI Service handler at [`climatextract/adapters/azure_openai.py`](https://github.com/gist-sustainability/climatextract/blob/main/climatextract/adapters/azure_openai.py) for legacy deployments. If neither fits your setup, you can write your own handler for any provider — see [Custom Providers](../user-guide/custom-providers.md).

Create a `.env` file in your working directory with the endpoint(s) for the handler you're using and the respective API key. `AZURE_AI_FOUNDRY_ENDPOINT` is required for the default Foundry handler; `AZURE_ENDPOINT` is required only if you opt into the Azure OpenAI Service handler.

```bash
AZURE_AI_FOUNDRY_ENDPOINT=https://your-foundry-endpoint.openai.azure.com/
AZURE_ENDPOINT=https://your-openai-endpoint.openai.azure.com/  # only if using the Azure OpenAI Service handler
API_KEY=your-api-key # you can also use personalized authentication workflows, see Step 4
API_VERSION=2024-12-01-preview
```

---

## Step 4 (optional): Configure personalized authentication to Azure

Instead of using an API_KEY (problem: different endpoints require different API_KEYs), you can also log in to Azure with your personal account.

Add to your `.env` file:

```bash
AZURE_USERNAME=your-username
AZURE_PASSWORD=your-password

# API_KEY=not-needed-anymore
```

This functionality is based on the [azure_authentication](https://github.com/soda-lmu/azure-auth-helper-python) package. Please refer to its [documentation](https://github.com/soda-lmu/azure-auth-helper-python) for alternative authentication workflows.

---

## Verify Installation

Test that everything is working:

```bash
python -c "from climatextract import extract; print('Installation successful!')"
```

---

## Next Steps

Ready to extract some data? Head to the [Quickstart](quickstart.md) guide.
