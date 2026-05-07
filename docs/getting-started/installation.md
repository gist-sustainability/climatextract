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

## Step 3: Configure Access to Large Language Models

You will need to set up acess to a Large Language Model and to an embedding model. You can either use our adapters that we have built for Microsoft Azure, or write your own adapter. Since our adapters are based on liteLLM, building your own should be straightforward — see [Custom Model Providers](../user-guide/custom-providers.md).

By default, the package will connect to Microsoft Azure's AI foundry.

### Using Azure's AI foundry

We commonly use Azure's AI foundry. Create an `.env` file in your working directory with the correct endpoint and the respective API key.

```bash
AZURE_AI_FOUNDRY_ENDPOINT=https://your-foundry-endpoint.openai.azure.com/
API_KEY=your-api-key # you can also use personalized authentication workflows, see Step 4
```

In the [configuration file](../user-guide/configuration.md) you specify which models you want to use, e.g.:

```bash
llm_model = "gpt-5-chat"
emb_model = "text-embedding-ada-002"
max_parallel_llm_prompts_running = 20
max_parallel_embedding_calls = 30
```

Make sure that models with these names have been deployed in your AI foundry instance.

Rate limit errors may occur depending on the quota you have assigned to each deployed model. Use the ``max_parallel_...``-parameters to limit the maximum number of API requests that will be sent in parallel, so that it matches with the model quota you have available.

The adapter for Azure's AI foundry is available at [`climatextract/adapters/azure_ai_foundry.py`](https://github.com/gist-sustainability/climatextract/blob/main/climatextract/adapters/azure_ai_foundry.py), useful for adaptions and debugging. A slightly different adapter / API endpoint may be needed if you wish to deploy models that are not from OpenAI.

### Using Azure OpenAI

For older models and legacy deployments, the package also ships an Azure OpenAI Service adapter at [`climatextract/adapters/azure_openai.py`](https://github.com/gist-sustainability/climatextract/blob/main/climatextract/adapters/azure_openai.py). The `.env` file when using this service should be as follows:

```bash
AZURE_ENDPOINT=https://your-openai-endpoint.openai.azure.com/
API_KEY=your-api-key # you can also use personalized authentication workflows, see Step 4
API_VERSION=2024-12-01-preview
```

---

## Step 4 (optional): Configure personalized authentication to Azure

Instead of using an API_KEY (problem: different endpoints require different API_KEYs), you can also log in to Azure with your personal account.

Install the `azure_authentication` package:

```bash
pip install "azure_authentication@git+https://github.com/soda-lmu/azure-auth-helper-python.git"
```

Add to your `.env` file:

```bash
AZURE_USERNAME=your-username
AZURE_PASSWORD=your-password

# API_KEY=not-needed-anymore
```

To facilitate login processes, you can use the following. Be aware of the risk if you store secret information on your hard drive.
```bash
AZURE_SODA_WEBLOGIN=advanced
AZURE_SODA_CREDENTIAL_PATH=my_secret_azure_credential.json
AZURE_SODA_ALLOW_UNENCRYPTED_STORAGE=True 
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

You may also want to set up [experiment tracking with MLflow](../user-guide/mlflow-setup.md) or start [sharing large (PDF) files](../user-guide/datalake-configuration.md) with team members via Azure Blob Storage.
