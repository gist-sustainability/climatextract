"""LiteLLM Gateway adapter for models served through a self-hosted LiteLLM proxy (OpenAI-compatible API).

On a self-hosted platform, create keys as follows. Insert your own <hostname>, <management-or-full-access-key> and the <unique-key-name> of the new key:
curl -s -X POST https://<hostname>/key/generate -H "Authorization: Bearer <management-or-full-access-key>" -H "Content-Type: application/json" -d '{"key_alias": "<unique-key-name", "max_budget": 1, "duration": "7mo", "models": ["text-embedding-ada-002", "gpt-6-luna", "gpt-6-sol", "DeepSeek-V4-Flash-0731", "gpt-oss-120b"], "allowed_routes": ["/chat/completions", "/embeddings", "/v1/models", "/key/info"]}' | jq -r | jq '{key_alias, key}'

Available model names are set by the gateway. It uses a community-maintained cost map.

Set the following environment variables:
``LITELLM_GATEWAY_API_BASE``
``LITELLM_GATEWAY_API_KEY``
"""

import os
import httpx
from typing import Optional, Tuple

import litellm
from litellm import EmbeddingResponse, ModelResponse

from climatextract.llm_embedding_api_bridge import (
    EmbeddingModelHandler,
    LlmHandler,
)

# Silently drop params a given model doesn't accept (e.g. ``temperature=0``
# on gpt-5.2, or ``logprobs`` on reasoning models). Recommended by
# LiteLLM for cross-provider code.
litellm.drop_params = True


def _api_base() -> str:
    base = os.getenv("LITELLM_GATEWAY_API_BASE")
    if not base:
        raise RuntimeError("Set LITELLM_GATEWAY_API_BASE to the gateway root.")
    return base.rstrip("/")


def _api_key() -> Optional[str]:
    return os.getenv("LITELLM_GATEWAY_API_KEY") or os.getenv("API_KEY")


def _common_model_dict(model_name: str) -> dict:
    d = {"model": f"litellm_proxy/{model_name}", "api_base": _api_base()}
    api_key = _api_key()
    if api_key:
        d["api_key"] = api_key
    d["num_retries"] = 8
    # Drop params serverside
    d["extra_body"] = {"drop_params": True}
    return d


def _cost(response) -> float:
    """USD cost the gateway reported for this call, else 0.0."""
    hidden = getattr(response, "_hidden_params", None) or {}
    headers = hidden.get("additional_headers") or {}
    try:
        # The gateway sets the cost in a custom header
        return float(headers.get("llm_provider-x-litellm-response-cost") or 0.0)
    except (TypeError, ValueError):
        pass
    try:
        # the client calculates the cost from the model's token usage and a cost map.
        return float(hidden.get("response_cost") or 0.0)
    except (TypeError, ValueError):  
        return 0.0
        

# ---------------------------------------------------------------------------
# LLM handler
# ---------------------------------------------------------------------------

class LiteLLMGatewayLlmHandler(LlmHandler):
    """LLM adapter for models served via a LiteLLM gateway.

    Extra keyword arguments are merged into the model dict last, so
    they override what TOML configured.
    """

    MODEL: str = "gpt-6-luna"

    def __init__(self, **extra_params):
        from climatextract import _runtime_config
        params = _runtime_config.get_current().llm_params

        self.model_name = params.llm_model or self.MODEL
        self.model_dict = _common_model_dict(self.model_name)
        self.model_dict["temperature"] = params.temperature
        if params.return_logprobs:
            self.model_dict["logprobs"] = True
        if params.reasoning_effort:
            self.model_dict["reasoning_effort"] = params.reasoning_effort
        self.model_dict.update(extra_params)
        self.max_concurrent_calls = params.max_parallel_llm_prompts_running or 1

    def get_model_dict(self) -> dict:
        return self.model_dict

    def get_max_concurrent_calls(self) -> int:
        return self.max_concurrent_calls

    def get_completion_and_cost(
        self, messages: list[dict]
    ) -> Tuple[ModelResponse, float]:
        response = litellm.completion(messages=messages, **self.model_dict)
        return response, _cost(response)

    async def aget_completion_and_cost(
        self, messages: list[dict]
    ) -> Tuple[ModelResponse, float]:
        response = await litellm.acompletion(messages=messages, **self.model_dict)
        return response, _cost(response)


# ---------------------------------------------------------------------------
# Embedding handler
# ---------------------------------------------------------------------------

class LiteLLMGatewayEmbeddingHandler(EmbeddingModelHandler):
    """Embedding adapter for models served via a LiteLLM gateway."""

    MODEL: str = "text-embedding-ada-002"

    def __init__(self):
        from climatextract import _runtime_config
        params = _runtime_config.get_current().semantic_search_params
        self.model_name = params.emb_model or self.MODEL
        self.model_dict = _common_model_dict(self.model_name)
        self.max_concurrent_calls = params.max_parallel_embedding_calls or 1

    def get_model_dict(self) -> dict:
        return self.model_dict

    def get_max_concurrent_calls(self) -> int:
        return self.max_concurrent_calls

    def get_embedding_and_cost(
        self, texts: list[str]
    ) -> Tuple[EmbeddingResponse, float]:
        response = litellm.embedding(input=texts, **self.model_dict)
        return response, _cost(response)

    async def aget_embedding_and_cost(
        self, texts: list[str]
    ) -> Tuple[EmbeddingResponse, float]:
        response = await litellm.aembedding(input=texts, **self.model_dict)
        return response, _cost(response)

# ---------------------------------------------------------------------------
# Account status
# ---------------------------------------------------------------------------

def list_models() -> list[str]:
    """Model aliases the gateway exposes to this key."""

    with httpx.Client(timeout=10.0) as client:
        r = client.get(f"{_api_base()}/v1/models",
                       headers={"Authorization": f"Bearer {_api_key()}"})
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]


def key_budget() -> dict:
    """Budget status of the configured key, from the gateway's ``/key/info``.

    ``budget_reset_at`` is only relevant if the budget resets periodically.
    """

    api_key = _api_key()
    if not api_key:
        raise RuntimeError(
            "No gateway key found: set LITELLM_GATEWAY_API_KEY (or API_KEY).")
    with httpx.Client(timeout=10.0) as client:
        r = client.get(f"{_api_base()}/key/info",
                       headers={"Authorization": f"Bearer {api_key}"})
        r.raise_for_status()
        info = r.json().get("info", {})

    spend = float(info.get("spend") or 0.0)
    max_budget = info.get("max_budget")
    return {
        "spend": spend,
        "max_budget": None if max_budget is None else float(max_budget),
        "remaining": None if max_budget is None else float(max_budget) - spend,
        "budget_expires": info.get('expires')[0:10],
        "budget_reset_at": info.get("budget_reset_at"),
    }


if __name__ == "__main__":
    LiteLLMGatewayLlmHandler().test_connection()
    LiteLLMGatewayEmbeddingHandler().test_connection()

    try:
        print("Key budget:", key_budget(), "\n")
    except Exception:
        print(f"Key budget unavailable.")
    print("Models on gateway:", list_models(), "\n")