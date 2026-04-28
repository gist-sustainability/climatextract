"""Shared utilities used by both Azure adapters.

Both ``azure_openai`` and ``azure_ai_foundry`` need the same Azure AD
auth fallback, the same cost-calculation safety wrapper, and the same
LiteLLM module-level configuration. Centralizing here keeps the two
adapter files focused on what's different (routing, env vars, model
quirks) instead of what's shared.
"""

import logging
from typing import Optional

import litellm

from climatextract.llm_embedding_api_bridge import ThreadSafeTokenProvider

logger = logging.getLogger(__name__)


# Silently drop params a given model doesn't accept (e.g. ``temperature=0``
# on gpt-5.2, or ``logprobs`` on reasoning models). Recommended by
# LiteLLM for cross-provider code.
litellm.drop_params = True

# Silence LiteLLM's stdout chatter (``Provider List:`` banner, success
# handler lines, completed-call logs). Our UsageCounter already captures
# tokens and cost.
litellm.suppress_debug_info = True
litellm.set_verbose = False
litellm.success_callback = []
litellm.failure_callback = []
for _logger_name in ("LiteLLM", "litellm", "litellm.utils", "litellm.main",
                     "litellm.cost_calculator", "litellm.proxy"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)


def azure_credential_provider() -> Optional[ThreadSafeTokenProvider]:
    """Return an Azure AD token provider if API_KEY is not set, else None.

    Used by both Azure OpenAI Service and Azure AI Foundry adapters when
    the user hasn't provided an explicit API key in env.
    """
    import os
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


def safe_cost(response) -> float:
    """Return the USD cost of a LiteLLM response, or 0.0 if unavailable.

    LiteLLM computes the cost during the call and stores it in
    ``response._hidden_params['response_cost']`` using the model name
    we sent. Re-computing via ``litellm.completion_cost(response)``
    looks at ``response.model``, which Azure populates with a
    server-side variant (e.g. ``gpt-5.2-chat-latest``) that isn't in
    LiteLLM's database. So we prefer the precomputed value and only
    fall back to recomputation if it's missing.
    """
    hidden = getattr(response, "_hidden_params", None) or {}
    precomputed = hidden.get("response_cost")
    if precomputed is not None:
        try:
            return float(precomputed)
        except (TypeError, ValueError):
            pass

    try:
        return float(litellm.completion_cost(completion_response=response) or 0.0)
    except Exception as e:
        logger.debug("completion_cost failed: %s", e)
        return 0.0


def find_litellm_canonical_key(deployment_name: str) -> Optional[str]:
    """Find the LiteLLM ``model_cost`` key that best matches a deployment.

    Resolution order:

    1. Exact bare match: ``azure/<name>`` or ``azure_ai/<name>``. LiteLLM
       maintains both for OpenAI-family models and these tend to be
       up-to-date with current production capabilities.
    2. Fall back to a dated variant (``azure/<name>-<YYYY-MM-DD>``)
       only when there's no bare match — this handles models that
       LiteLLM only catalogues by version.

    Returns ``None`` if LiteLLM has no entry matching the deployment.
    """
    for prefix in ("azure", "azure_ai"):
        bare = f"{prefix}/{deployment_name}"
        if bare in litellm.model_cost:
            return bare

    for prefix in ("azure", "azure_ai"):
        dated_prefix = f"{prefix}/{deployment_name}-"
        matches = [k for k in litellm.model_cost if k.startswith(dated_prefix)]
        if matches:
            return max(matches)

    return None


def ensure_litellm_metadata_registered(deployment_name: str, our_prefix: str = "azure") -> None:
    """Make sure LiteLLM has metadata registered under our deployment's
    full key (``<our_prefix>/<deployment_name>``).

    The case this matters for: LiteLLM lists most OpenAI-family models
    under both ``azure/`` and ``azure_ai/`` prefixes, but lists OSS-only
    models (Llama, gpt-oss-120b) ONLY under ``azure_ai/``. When we call
    such a model with the legacy ``azure/`` path (because our Foundry
    resource exposes the legacy API surface), LiteLLM's cost calculator
    can't find pricing under that key and reports $0.

    This function copies metadata from whatever canonical entry
    LiteLLM has, overriding ``litellm_provider`` to match our prefix —
    otherwise LiteLLM's cost calculator sees prefix/provider mismatch
    and bails out.

    Idempotent — safe to call from every handler ``__init__``.
    """
    target_key = f"{our_prefix}/{deployment_name}"
    if target_key in litellm.model_cost:
        return

    canonical = find_litellm_canonical_key(deployment_name)
    if canonical is None:
        logger.debug(
            "No LiteLLM metadata found for deployment '%s'; "
            "cost calculation will not be active for this model.",
            deployment_name,
        )
        return

    try:
        info = litellm.get_model_info(canonical)
    except Exception as e:
        logger.debug("get_model_info(%r) failed: %s", canonical, e)
        return

    if not info:
        return

    info_for_us = dict(info)
    info_for_us["litellm_provider"] = our_prefix

    try:
        litellm.register_model({target_key: info_for_us})
    except Exception as e:  # pragma: no cover
        logger.debug("register_model failed for %r: %s", target_key, e)
