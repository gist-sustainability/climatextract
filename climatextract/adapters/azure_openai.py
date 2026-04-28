"""Azure OpenAI Service adapter.

For models deployed on the original Azure OpenAI Service (the product
that predates Azure AI Foundry). Goes through LiteLLM's ``azure/``
provider with ``api_version`` and the ``AZURE_ENDPOINT`` env var.

If your models live on Azure AI Foundry instead, use
``climatextract.adapters.azure_ai_foundry``.
"""

import os
from typing import Tuple

import litellm
from litellm import EmbeddingResponse, ModelResponse

from climatextract.adapters._shared import (
    azure_credential_provider,
    ensure_litellm_metadata_registered,
    safe_cost,
)
from climatextract.llm_embedding_api_bridge import (
    EmbeddingModelHandler,
    LlmHandler,
)


def _api_version() -> str:
    return os.environ.get("API_VERSION", "2024-12-01-preview")


def _common_model_dict(model_name: str) -> dict:
    """Build the auth + endpoint piece of model_dict that's identical
    for both LLM and embedding handlers on Azure OpenAI Service."""
    d = {
        "model": f"azure/{model_name}",
        "api_base": os.environ["AZURE_ENDPOINT"],
        "api_version": _api_version(),
    }
    api_key = os.getenv("API_KEY")
    if api_key:
        d["api_key"] = api_key
    else:
        provider = azure_credential_provider()
        if provider is not None:
            d["azure_ad_token_provider"] = provider
    return d


# ---------------------------------------------------------------------------
# LLM handler
# ---------------------------------------------------------------------------

class AzureOpenAILlmHandler(LlmHandler):
    """LiteLLM-backed LLM adapter for Azure OpenAI Service deployments."""

    def __init__(self):
        from climatextract import _runtime_config
        params = _runtime_config.get_current().llm_params

        self.model_name = params.llm_model or self.MODEL
        ensure_litellm_metadata_registered(self.model_name, our_prefix="azure")
        self.model_dict = _common_model_dict(self.model_name)
        self.model_dict["temperature"] = params.temperature
        if params.return_logprobs:
            self.model_dict["logprobs"] = True
        if params.reasoning_effort:
            self.model_dict["reasoning_effort"] = params.reasoning_effort

    def get_model_dict(self) -> dict:
        return self.model_dict

    def get_completion_and_cost(
        self, messages: list[dict]
    ) -> Tuple[ModelResponse, float]:
        response = litellm.completion(messages=messages, **self.model_dict)
        return response, safe_cost(response)

    async def aget_completion_and_cost(
        self, messages: list[dict]
    ) -> Tuple[ModelResponse, float]:
        response = await litellm.acompletion(messages=messages, **self.model_dict)
        return response, safe_cost(response)


# ---------------------------------------------------------------------------
# Embedding handler
# ---------------------------------------------------------------------------

class AzureOpenAIEmbeddingHandler(EmbeddingModelHandler):
    """LiteLLM-backed embedding adapter for Azure OpenAI Service."""

    def __init__(self):
        from climatextract import _runtime_config
        params = _runtime_config.get_current().semantic_search_params
        self.model_name = params.emb_model or self.MODEL
        ensure_litellm_metadata_registered(self.model_name, our_prefix="azure")
        self.model_dict = _common_model_dict(self.model_name)

    def get_model_dict(self) -> dict:
        return self.model_dict

    def get_embedding_and_cost(
        self, texts: list[str]
    ) -> Tuple[EmbeddingResponse, float]:
        texts = ["empty parsed page" if t == "" else t for t in texts]
        response = litellm.embedding(input=texts, **self.model_dict)
        return response, safe_cost(response)

    async def aget_embedding_and_cost(
        self, texts: list[str]
    ) -> Tuple[EmbeddingResponse, float]:
        texts = ["empty parsed page" if t == "" else t for t in texts]
        response = await litellm.aembedding(input=texts, **self.model_dict)
        return response, safe_cost(response)
