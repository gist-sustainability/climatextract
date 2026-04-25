"""Azure OpenAI + Azure AI Foundry adapters built on LiteLLM.

Routes bare model names (``gpt-4o-mini-2024-07-18``, ``gpt-5.2-chat-2025-12-11``,
``text-embedding-3-large``, ...) to the right LiteLLM prefix (``azure/`` vs
``azure_ai/``) and endpoint, registers custom pricing for Foundry models that
aren't in LiteLLM's community cost map, and handles per-model quirks
(``reasoning_effort``, ``supports_logprobs``, ``temperature``).
"""

import logging
import os
from typing import Optional, Tuple

import litellm
from litellm import EmbeddingResponse, ModelResponse

from climatextract.llm_embedding_api_bridge import (
    EmbeddingModelHandler,
    LlmHandler,
    ThreadSafeTokenProvider,
)

logger = logging.getLogger(__name__)

# Silently drop params that a given model doesn't accept (e.g.
# ``temperature=0`` on gpt-5.2, or ``logprobs`` on reasoning models).
# Documented behavior, recommended by LiteLLM for cross-provider code:
# https://docs.litellm.ai/docs/completion/drop_params
litellm.drop_params = True

# Silence LiteLLM's stdout chatter (``Provider List:`` banner, success
# handler lines, completed-call logs). None of it is informative for our
# pipeline — our UsageCounter already captures tokens and cost.
litellm.suppress_debug_info = True
litellm.set_verbose = False
litellm.success_callback = []
litellm.failure_callback = []
for _logger_name in ("LiteLLM", "litellm", "litellm.utils", "litellm.main",
                     "litellm.cost_calculator", "litellm.proxy"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------

# Models that live on Azure AI Foundry (``azure_ai/`` prefix, Foundry
# endpoint). Everything else is assumed to be on regular Azure OpenAI
# (``azure/`` prefix, Azure OpenAI endpoint).
_FOUNDRY_MODELS = {
    "gpt-oss-120b",
    "gpt-4.1-2025-04-14",
    "gpt-5-chat-2025-08-07",
    "gpt-5.2-chat-2025-12-11",
    "Llama-4-Maverick-17B-128E-Instruct-FP8",
}

# Per-model ``reasoning_effort`` defaults.
# Users who want a different effort can pass
# ``reasoning_effort`` explicitly to the handler constructor.
_DEFAULT_REASONING_EFFORT = {
    "gpt-5.2-chat-2025-12-11": "none",
}

# Per-model concurrent-call fallbacks. Used only when neither the
# constructor nor the TOML config specifies ``max_concurrent_calls`` /
# ``max_parallel_llm_prompts_running``.
_DEFAULT_MAX_CONCURRENT = {
    "gpt-35-turbo-16k": 8,
    "gpt-4o-mini-2024-07-18": 25,
    "gpt-4o-2024-11-20": 25,
    "gpt-oss-120b": 25,
    "gpt-4.1-2025-04-14": 4,
    "gpt-5-chat-2025-08-07": 8,
    "gpt-5.2-chat-2025-12-11": 8,
    "Llama-4-Maverick-17B-128E-Instruct-FP8": 25,
    "o3-mini-2025-01-31": 2,
    "text-embedding-ada-002": 5,
    "text-embedding-3-large": 10,
}

# USD price overrides for models LiteLLM's community cost map doesn't
# know about (as of 2026-04). Applied via ``litellm.register_model`` at
# import time so ``litellm.completion_cost(response)`` returns correct
# values for these models.
_CUSTOM_PRICING = {
    "azure_ai/Llama-4-Maverick-17B-128E-Instruct-FP8": {
        "input_cost_per_token": 0.000000303,
        "output_cost_per_token": 0.00000121,
    },
    "azure_ai/gpt-5.2-chat-2025-12-11": {
        "input_cost_per_token": 0.00000175,
        "output_cost_per_token": 0.00001400,
    },
    "azure_ai/gpt-5-chat-2025-08-07": {
        "input_cost_per_token": 0.00000125,
        "output_cost_per_token": 0.00001000,
    },
    "azure_ai/gpt-oss-120b": {
        "input_cost_per_token": 0.0000003,
        "output_cost_per_token": 0.0000025,
    },
    "azure_ai/gpt-4.1-2025-04-14": {
        "input_cost_per_token": 0.000002,
        "output_cost_per_token": 0.000008,
    },
}

try:
    litellm.register_model(_CUSTOM_PRICING)
except Exception as e:  # pragma: no cover - defensive
    logger.warning("Could not register custom LiteLLM pricing: %s", e)


def _azure_api_version() -> str:
    return os.environ.get("API_VERSION", "2024-12-01-preview")


def _azure_endpoint() -> str:
    return os.environ["AZURE_ENDPOINT"]


def _foundry_endpoint() -> str:
    # LiteLLM's ``azure_ai/`` provider wants the OpenAI-compatible base
    # path (``.../openai/v1``), which differs from the raw Foundry URL.
    base = os.environ["AZURE_AI_FOUNDRY_ENDPOINT"].rstrip("/")
    if not base.endswith("/openai/v1"):
        base = base + "/openai/v1"
    return base


def _azure_credential_provider() -> Optional[ThreadSafeTokenProvider]:
    """Return an Azure AD token provider if API_KEY is not set, else None."""
    if os.getenv("API_KEY"):
        return None
    try:
        from azure_authentication import customized_azure_login  # type: ignore
        credential = customized_azure_login.CredentialFactory().select_credential()
        return ThreadSafeTokenProvider(credential)
    except Exception as e:  # pragma: no cover
        logger.warning(
            "Azure AD auth unavailable and API_KEY not set: %s", e)
        return None


def _route(model_name: str) -> Tuple[str, str]:
    """Return (litellm_model_string, api_base) for the given bare model."""
    if model_name in _FOUNDRY_MODELS:
        return f"azure_ai/{model_name}", _foundry_endpoint()
    return f"azure/{model_name}", _azure_endpoint()


# ---------------------------------------------------------------------------
# LLM handler
# ---------------------------------------------------------------------------

class AzureOpenAILlmHandler(LlmHandler):
    """LiteLLM-backed LLM adapter for Azure OpenAI + Azure AI Foundry.

    Constructor accepts a bare model name (``gpt-5.2-chat-2025-12-11``);
    endpoint routing, auth, and per-model quirks are inferred internally.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        return_logprobs: bool = True,
        reasoning_effort: Optional[str] = None,
        max_concurrent_calls: Optional[int] = None,
    ):
        self.model_name = model
        litellm_model, api_base = _route(model)

        self.model_dict: dict = {
            "model": litellm_model,
            "api_base": api_base,
        }

        # Regular Azure OpenAI wants api_version; Azure AI Foundry does not.
        if litellm_model.startswith("azure/"):
            self.model_dict["api_version"] = _azure_api_version()

        # Auth: API key if available, else Azure AD token provider.
        api_key = os.getenv("API_KEY")
        if api_key:
            self.model_dict["api_key"] = api_key
        else:
            provider = _azure_credential_provider()
            if provider is not None:
                self.model_dict["azure_ad_token_provider"] = provider

        # Temperature: always pass through. ``drop_params=True`` will drop
        # it for models that reject it (e.g. gpt-5.2), leaving Azure to
        # apply its own default.
        self.model_dict["temperature"] = temperature

        # Logprobs: request them if the user asked. Models that reject the
        # param (reasoning models, gpt-5.2) will have it silently dropped
        # by LiteLLM because ``drop_params=True`` is set at module load.
        if return_logprobs:
            self.model_dict["logprobs"] = True

        # Reasoning effort: user-supplied takes priority; otherwise use
        # the per-model default if one is registered.
        effective_effort = (
            reasoning_effort
            if reasoning_effort is not None
            else _DEFAULT_REASONING_EFFORT.get(model)
        )
        if effective_effort is not None:
            self.model_dict["reasoning_effort"] = effective_effort

        # Concurrency cap.
        if max_concurrent_calls is not None:
            self.max_concurrent_calls = max_concurrent_calls
        else:
            self.max_concurrent_calls = _DEFAULT_MAX_CONCURRENT.get(model, 4)

    def get_model_dict(self) -> dict:
        return self.model_dict

    def get_max_concurrent_calls(self) -> int:
        return self.max_concurrent_calls

    def get_completion_and_cost(
        self, messages: list[dict]
    ) -> Tuple[ModelResponse, float]:
        response = litellm.completion(messages=messages, **self.model_dict)
        cost = _safe_cost(response)
        return response, cost

    async def aget_completion_and_cost(
        self, messages: list[dict]
    ) -> Tuple[ModelResponse, float]:
        response = await litellm.acompletion(messages=messages, **self.model_dict)
        cost = _safe_cost(response)
        return response, cost


# ---------------------------------------------------------------------------
# Embedding handler
# ---------------------------------------------------------------------------

class AzureOpenAIEmbeddingHandler(EmbeddingModelHandler):
    """LiteLLM-backed embedding adapter for Azure OpenAI."""

    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        max_concurrent_calls: Optional[int] = None,
    ):
        self.model_name = model
        litellm_model, api_base = _route(model)

        self.model_dict: dict = {
            "model": litellm_model,
            "api_base": api_base,
        }
        if litellm_model.startswith("azure/"):
            self.model_dict["api_version"] = _azure_api_version()

        api_key = os.getenv("API_KEY")
        if api_key:
            self.model_dict["api_key"] = api_key
        else:
            provider = _azure_credential_provider()
            if provider is not None:
                self.model_dict["azure_ad_token_provider"] = provider

        if max_concurrent_calls is not None:
            self.max_concurrent_calls = max_concurrent_calls
        else:
            self.max_concurrent_calls = _DEFAULT_MAX_CONCURRENT.get(model, 2)

    def get_model_dict(self) -> dict:
        return self.model_dict

    def get_max_concurrent_calls(self) -> int:
        return self.max_concurrent_calls

    def get_embedding_and_cost(
        self, texts: list[str]
    ) -> Tuple[EmbeddingResponse, float]:
        # Guard against empty-string inputs (Azure returns 400).
        texts = ["empty parsed page" if t == "" else t for t in texts]
        response = litellm.embedding(input=texts, **self.model_dict)
        cost = _safe_cost(response)
        return response, cost

    async def aget_embedding_and_cost(
        self, texts: list[str]
    ) -> Tuple[EmbeddingResponse, float]:
        texts = ["empty parsed page" if t == "" else t for t in texts]
        response = await litellm.aembedding(input=texts, **self.model_dict)
        cost = _safe_cost(response)
        return response, cost


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_cost(response) -> float:
    """Return ``completion_cost`` or 0.0 if LiteLLM can't price this model.

    Missing cost data (common for brand-new Foundry models that aren't in
    ``_CUSTOM_PRICING`` yet) shouldn't bring down a whole extraction run.
    """
    try:
        return float(litellm.completion_cost(completion_response=response) or 0.0)
    except Exception as e:
        logger.debug("completion_cost failed: %s", e)
        return 0.0
