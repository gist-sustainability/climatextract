"""Azure AI Foundry adapter.

For models deployed via Azure AI Foundry. Foundry resources typically
expose two API surfaces:

  - the OpenAI-compatible ``/openai/v1/...`` path (newer)
  - the legacy Azure OpenAI Service ``/openai/deployments/{name}/...?api-version=X`` path

LiteLLM's ``azure_ai/`` provider targets the v1 path, but as of LiteLLM
1.83 it has a known bug in URL construction (multiple open issues:
github.com/BerriAI/litellm/issues/16449, /16577, /17765, /7275, /12945)
that produces 404s against Foundry endpoints. Direct curl against the
same v1 endpoint works fine — the bug is in LiteLLM's path-append
logic.

The recommended workaround (also our approach): use LiteLLM's ``azure/``
provider against the legacy URL surface. This works on any Foundry
resource that has legacy backward-compat enabled (the default for most
resources). Pointing ``api_base`` at the Foundry endpoint env var keeps
us aimed at the right resource.

If a Foundry resource is configured to expose ONLY the v1 surface, this
adapter won't reach it. In that case the LiteLLM bug needs to be fixed
upstream, or a custom URL-construction adapter would be needed.
"""

import os
from typing import Tuple

import litellm
from litellm import EmbeddingResponse, ModelResponse

from climatextract.adapters._shared import (
    azure_credential_provider,
    ensure_litellm_metadata_registered,
    find_litellm_canonical_key,
    safe_cost,
)
from climatextract.llm_embedding_api_bridge import (
    EmbeddingModelHandler,
    LlmHandler,
)


# Foundry models exposed via the legacy Azure path require a recent
# api-version because some (o3-mini, gpt-5.2-chat) need 2024-12-01 or
# later. Newer api-versions are backward-compatible.
_FOUNDRY_API_VERSION = "2025-01-01-preview"


def _foundry_endpoint() -> str:
    """Foundry resource root URL (no path suffix)."""
    return os.environ["AZURE_AI_FOUNDRY_ENDPOINT"].rstrip("/")


def _common_model_dict(model_name: str) -> dict:
    """Build the auth + endpoint piece of model_dict shared by both
    LLM and embedding handlers on Foundry."""
    d = {
        "model": f"azure/{model_name}",
        "api_base": _foundry_endpoint(),
        "api_version": _FOUNDRY_API_VERSION,
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

class AzureAIFoundryLlmHandler(LlmHandler):
    """LiteLLM-backed LLM adapter for Azure AI Foundry deployments."""

    MODEL = "gpt-5-nano"

    def __init__(self):
        from climatextract import _runtime_config
        params = _runtime_config.get_current().llm_params

        self.model_name = params.llm_model or self.MODEL
        ensure_litellm_metadata_registered(self.model_name, our_prefix="azure")

        self.model_dict = _common_model_dict(self.model_name)
        if not self._should_skip_temperature_and_logprobs():
            self.model_dict["temperature"] = params.temperature
            if params.return_logprobs:
                self.model_dict["logprobs"] = True
        if params.reasoning_effort:
            self.model_dict["reasoning_effort"] = params.reasoning_effort

    def _should_skip_temperature_and_logprobs(self) -> bool:
        # Azure's gpt-5.2-chat (and other reasoning-flagged models)
        # rejects temperature=0 and logprobs. LiteLLM's drop_params
        # doesn't catch this — see the github.com/BerriAI/litellm
        # issues referenced at the top of this file. We use LiteLLM's
        # ``supports_reasoning`` flag as the heuristic.
        canonical = find_litellm_canonical_key(self.model_name)
        return bool(canonical and litellm.supports_reasoning(model=canonical))

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

class AzureAIFoundryEmbeddingHandler(EmbeddingModelHandler):
    """LiteLLM-backed embedding adapter for Azure AI Foundry. Use only
    if your embedding model is deployed on Foundry; otherwise pair the
    Foundry LLM handler with ``AzureOpenAIEmbeddingHandler``."""

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
