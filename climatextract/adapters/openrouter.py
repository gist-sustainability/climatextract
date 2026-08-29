"""OpenRouter Service adapter.

For models served through OpenRouter (https://openrouter.ai), an
OpenAI-compatible gateway in front of many providers. Goes through
LiteLLM's ``openrouter/`` provider.

Check the following links to understand routing and budgeting with OpenRouter:
- https://openrouter.ai/models
- https://openrouter.ai/docs/api_reference/limits
- https://openrouter.ai/docs/guides/routing/provider-selection
- https://openrouter.ai/docs/guides/routing/model-variants/floor

Model names use OpenRouter's ``<vendor>/<model>`` ids, e.g.
``openai/gpt-4o-mini``, ``anthropic/claude-sonnet-4`` or
``qwen/qwen3-embedding-8b`` — see https://openrouter.ai/models
(and https://openrouter.ai/api/v1/embeddings/models for the
embedding catalogue).

Authentication: set the ``OPENROUTER_API_KEY`` env var (``API_KEY``
also works, for symmetry with the Azure adapters). To target a
self-hosted or proxied OpenRouter instance, set ``OPENROUTER_BASE_URL``
(or LiteLLM's own ``OPENROUTER_API_BASE``); otherwise the public
``https://openrouter.ai/api/v1`` is used.

The two handlers source their cost differently, each in its own
``_cost``: completions read what OpenRouter billed, embeddings have
LiteLLM compute it. See those methods for why.
"""

import os
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
    """OpenRouter endpoint, overridable to reach a proxied instance."""
    return (os.getenv("OPENROUTER_BASE_URL")
            or os.getenv("OPENROUTER_API_BASE")
            or "https://openrouter.ai/api/v1").rstrip("/")


def _api_key() -> Optional[str]:
    return os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEY")


def _common_model_dict(model_name: str) -> dict:
    """Build the auth + routing piece of model_dict that's identical
    for both LLM and embedding handlers on OpenRouter."""
    d = {"model": f"openrouter/{model_name}", "api_base": _api_base()}
    api_key = _api_key()
    if api_key:
        d["api_key"] = api_key
    # else: LiteLLM looks up OPENROUTER_API_KEY / OR_API_KEY itself and
    # raises a clear AuthenticationError if none is set.
    d["num_retries"] = 8
    return d


# ---------------------------------------------------------------------------
# LLM handler
# ---------------------------------------------------------------------------

class OpenRouterLlmHandler(LlmHandler):
    """LiteLLM-backed LLM adapter for models served via OpenRouter.

    Extra keyword arguments are merged into the model dict and passed
    straight to ``litellm.completion`` / ``acompletion``. They are
    applied last, so they also override what TOML configured::

        OpenRouterLlmHandler(max_tokens=2000, seed=42)

    OpenRouter-specific routing options that LiteLLM has no named
    parameter for travel in ``extra_body``::

        OpenRouterLlmHandler(extra_body={
            "provider": {"only": ["deepinfra"], "sort": "price"}})

    See https://openrouter.ai/docs/guides/routing/provider-selection.
    """

    MODEL: str = "openai/gpt-4o-mini"

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
        # Caller-supplied params win over the TOML-derived defaults.
        self.model_dict.update(extra_params)
        self.max_concurrent_calls = params.max_parallel_llm_prompts_running or 1

    def get_model_dict(self) -> dict:
        return self.model_dict

    def get_max_concurrent_calls(self) -> int:
        return self.max_concurrent_calls

    @staticmethod
    def _cost(response: ModelResponse) -> float:
        """USD OpenRouter billed for the call, from ``usage.cost``.

        LiteLLM asks for this by adding ``usage: {"include": true}`` to
        every OpenRouter chat request. It is what the provider the
        request was routed to actually charged, which can be below the
        headline per-token rate, so it beats pricing the call from a
        static catalogue. 0.0 if a proxied endpoint omits the field.
        """
        return float(
            getattr(getattr(response, "usage", None), "cost", 0.0) or 0.0)

    def get_completion_and_cost(
        self, messages: list[dict]
    ) -> Tuple[ModelResponse, float]:
        response = litellm.completion(messages=messages, **self.model_dict)
        return response, self._cost(response)

    async def aget_completion_and_cost(
        self, messages: list[dict]
    ) -> Tuple[ModelResponse, float]:
        response = await litellm.acompletion(messages=messages, **self.model_dict)
        return response, self._cost(response)


# ---------------------------------------------------------------------------
# Embedding handler
# ---------------------------------------------------------------------------

class OpenRouterEmbeddingHandler(EmbeddingModelHandler):
    """LiteLLM-backed embedding adapter for OpenRouter.

    OpenRouter serves only a subset of models on its ``/embeddings``
    endpoint — see https://openrouter.ai/api/v1/embeddings/models.
    """

    MODEL: str = "openai/text-embedding-ada-002"

    def __init__(self):
        from climatextract import _runtime_config
        params = _runtime_config.get_current().semantic_search_params
        self.model_name = params.emb_model or self.MODEL
        self.model_dict = _common_model_dict(self.model_name)
        self._register_price()
        self.max_concurrent_calls = params.max_parallel_embedding_calls or 1

    def _register_price(self) -> None:
        """Teach LiteLLM what OpenRouter charges to embed with this model.

        LiteLLM's pricing catalogue has no ``openrouter/`` embedding
        entries at all, which is why it otherwise reports $0.00 for
        every embedding call. Registering the per-token price from
        OpenRouter's public catalogue lets LiteLLM price the call
        inline, which is what ``_cost`` then reads.

        Best-effort and idempotent: if the catalogue can't be reached or
        doesn't list the model, costs simply stay at $0.00.
        """
        import logging
        import httpx

        key = f"openrouter/{self.model_name}"
        if key in litellm.model_cost:
            return

        api_base = self.model_dict["api_base"]
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{api_base}/embeddings/models")
                response.raise_for_status()
                catalogue = response.json().get("data", [])
            price = next(
                float(m["pricing"]["prompt"])
                for m in catalogue if m.get("id") == self.model_name
            )
        except Exception as e:
            logging.getLogger(__name__).debug(
                "No OpenRouter price for embedding model %r (%s); "
                "embedding costs will report 0.0.",
                self.model_name, type(e).__name__)
            return

        litellm.register_model({key: {
            "input_cost_per_token": price,
            "output_cost_per_token": 0.0,
            "litellm_provider": "openrouter",
            "mode": "embedding",
        }})

    def get_model_dict(self) -> dict:
        return self.model_dict

    def get_max_concurrent_calls(self) -> int:
        return self.max_concurrent_calls

    @staticmethod
    def _cost(response: EmbeddingResponse) -> float:
        """USD cost LiteLLM computed from the price ``_register_price`` set.

        Unlike completions this is not read off the response: OpenRouter
        does report what it billed, but LiteLLM's embedding parser
        discards that field (along with ``cost_details``, ``id`` and
        ``provider``) and keeps no copy, so it cannot be read back.
        """
        hidden = getattr(response, "_hidden_params", None) or {}
        return float(hidden.get("response_cost") or 0.0)

    def get_embedding_and_cost(
        self, texts: list[str]
    ) -> Tuple[EmbeddingResponse, float]:
        response = litellm.embedding(input=texts, **self.model_dict)
        return response, self._cost(response)

    async def aget_embedding_and_cost(
        self, texts: list[str]
    ) -> Tuple[EmbeddingResponse, float]:
        response = await litellm.aembedding(input=texts, **self.model_dict)
        return response, self._cost(response)



# ---------------------------------------------------------------------------
# Account status
# ---------------------------------------------------------------------------

class OpenRouterKeyInfo:
    """Credit and usage status for the configured OpenRouter API key.

    Wraps https://openrouter.ai/api/v1/key, which reports spend to date,
    any credit limit set on the key, and whether it is on the free tier.
    Handy for checking budget before a long extraction run — it is not a
    handler and nothing in the pipeline calls it.
    """

    def fetch(self) -> dict:
        """Return the endpoint's ``data`` payload.

        Unlike ``_register_price`` this does not degrade quietly: a
        missing or rejected key raises, since finding that out is the
        whole point of asking.
        """
        import httpx

        api_key = _api_key()
        if not api_key:
            raise RuntimeError(
                "No OpenRouter key found: set OPENROUTER_API_KEY (or API_KEY).")
        with httpx.Client(timeout=10.0) as client:
            response = client.get(
                f"{_api_base()}/key",
                headers={"Authorization": f"Bearer {api_key}"})
            response.raise_for_status()
            return response.json().get("data", {})

    def print_status(self) -> None:
        """Print the key's credit limit and usage, in the spirit of the
        handlers' ``test_connection``.

        Usage counters are cumulative over the current UTC day, week
        (starting Monday) and month; ``usage`` is all-time.
        """
        data = self.fetch()

        def usd(value) -> str:
            """``None`` means unlimited on the limit fields."""
            return "unlimited" if value is None else f"${float(value):.4f}"

        print(f"OpenRouter key {data.get('label')} at {_api_base()}")
        print(f"  credit limit         : {usd(data.get('limit'))}")
        print(f"  limit remaining      : {usd(data.get('limit_remaining'))}")
        print(f"  limit resets         : {data.get('limit_reset') or 'never'}")
        print(f"  BYOK counts to limit : {data.get('include_byok_in_limit')}")
        print(f"  free tier            : {data.get('is_free_tier')}")
        print(f"  used all time        : {usd(data.get('usage') or 0.0)}")
        print(f"  used today (UTC)     : {usd(data.get('usage_daily') or 0.0)}")
        print(f"  used this week (Mon) : {usd(data.get('usage_weekly') or 0.0)}")
        print(f"  used this month      : {usd(data.get('usage_monthly') or 0.0)}")


if __name__ == "__main__":

    OpenRouterKeyInfo().print_status()
    OpenRouterEmbeddingHandler().test_connection()
    OpenRouterLlmHandler().test_connection()
