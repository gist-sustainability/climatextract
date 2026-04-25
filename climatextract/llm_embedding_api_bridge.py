"""Provider-agnostic abstractions for LLM and embedding calls.

Users inject a handler (an ``EmbeddingModelHandler`` / ``LlmHandler`` subclass)
that knows how to talk to a specific provider. The package-side wrappers
(``EmbeddingModel`` / ``Llm``) add bookkeeping (usage, cost, concurrency
semaphore) on top of whatever the handler does.

The handlers in ``climatextract.adapters.*`` use LiteLLM under the hood to
reach Azure OpenAI, Azure AI Foundry, and (with user-written adapters) any
other provider LiteLLM supports. The package itself does not depend on
LiteLLM except for the ``EmbeddingResponse`` / ``ModelResponse`` return
types declared on the handler ABCs.
"""

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

from litellm import EmbeddingResponse, ModelResponse

logger = logging.getLogger(__name__)


class UsageCounter:
    """Accumulates token counts and USD cost across calls.

    Populated from provider responses (``response.usage`` + LiteLLM's
    ``completion_cost``); the package's wrapper classes push into it after
    every call.
    """

    def __init__(self):
        self.reset_counts()

    def reset_counts(self):
        self.prompt_llm_token_count = 0
        self.completion_llm_token_count = 0
        self.total_llm_token_count = 0
        self.total_embedding_token_count = 0
        self.total_cost = 0.0

    def get_usage_dict(self) -> dict:
        return {
            "embedding_tokens": self.total_embedding_token_count,
            "llm_prompt_tokens": self.prompt_llm_token_count,
            "llm_completion_tokens": self.completion_llm_token_count,
            "total_llm_token_count": self.total_llm_token_count,
            "total_cost_in_dollar": self.total_cost,
        }

    def add_prompt_tokens(self, count: int):
        self.prompt_llm_token_count += count
        self.total_llm_token_count += count

    def add_completion_tokens(self, count: int):
        self.completion_llm_token_count += count
        self.total_llm_token_count += count

    def add_embedding_tokens(self, count: int):
        self.total_embedding_token_count += count

    def add_cost(self, cost: float):
        self.total_cost += float(cost or 0.0)


class ThreadSafeTokenProvider:
    """Caches an Azure AD token and refreshes it when it's within two
    minutes of expiring. Kept here (rather than in an adapter) so that
    user-written Azure adapters can import it without pulling the whole
    adapter module in.
    """

    def __init__(self, credential, scope="https://cognitiveservices.azure.com/.default"):
        self.credential = credential
        self.scope = scope
        self.token = None
        self.expires_on = 0
        self.lock = threading.Lock()

    def __call__(self):
        # Renew Token, if it expires in <2min
        if self.token is None or (self.expires_on - time.time() < 120):
            with self.lock:
                # Check again in Lock
                if self.token is None or (self.expires_on - time.time() < 120):
                    t = self.credential.get_token(self.scope)
                    self.token = t.token
                    self.expires_on = t.expires_on
        return self.token


# ---------------------------------------------------------------------------
# Embedding side
# ---------------------------------------------------------------------------

class EmbeddingModelHandler(ABC):
    """Interface a user-written embedding adapter must satisfy."""

    @abstractmethod
    def get_embedding_and_cost(self, texts: list[str]) -> Tuple[EmbeddingResponse, float]: ...

    @abstractmethod
    async def aget_embedding_and_cost(self, texts: list[str]) -> Tuple[EmbeddingResponse, float]: ...

    @abstractmethod
    def get_model_dict(self) -> dict: ...

    @abstractmethod
    def get_max_concurrent_calls(self) -> int: ...


def _extract_vectors(response: EmbeddingResponse) -> list[list[float]]:
    """Return ``list[list[float]]`` regardless of whether items in
    ``response.data`` are dicts or objects (LiteLLM uses dict-like objects
    that support both forms)."""
    vectors = []
    for item in response.data:
        if isinstance(item, dict):
            vectors.append(item["embedding"])
        else:
            vectors.append(item.embedding)
    return vectors


class EmbeddingModel:
    """Package-side embedding wrapper.

    Wraps a user-supplied ``EmbeddingModelHandler`` with usage accounting
    and a semaphore. Pipeline code calls this class, not the handler.
    """

    def __init__(self, embedding_handler: EmbeddingModelHandler,
                 usage_counter: Optional[UsageCounter] = None):
        self.embedding_handler = embedding_handler
        self.usage_counter = usage_counter if usage_counter is not None else UsageCounter()
        self.embedding_semaphore = asyncio.Semaphore(
            embedding_handler.get_max_concurrent_calls())
        # Cached once to make repr() and get_embed_dimension() cheap.
        self._cached_model_name: Optional[str] = None
        self._cached_embed_dimension: Optional[int] = None

    def __repr__(self) -> str:
        return f"EmbeddingModel(model_name={self.get_model_name()!r})"

    @property
    def token_counter(self) -> UsageCounter:
        """Alias for ``usage_counter`` — same object, different spelling."""
        return self.usage_counter

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Sync embedding call. Returns ``list[list[float]]``."""
        response, cost = self.embedding_handler.get_embedding_and_cost(texts)
        self._record_usage(response, cost)
        return _extract_vectors(response)

    async def aget_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Async embedding call with concurrency control."""
        async with self.embedding_semaphore:
            response, cost = await self.embedding_handler.aget_embedding_and_cost(texts)
        self._record_usage(response, cost)
        return _extract_vectors(response)

    def _record_usage(self, response: EmbeddingResponse, cost: float) -> None:
        try:
            prompt_tokens = response.usage.prompt_tokens
        except AttributeError:
            prompt_tokens = 0
        self.usage_counter.add_embedding_tokens(prompt_tokens)
        self.usage_counter.add_cost(cost)

    def get_model_name(self) -> str:
        if self._cached_model_name is not None:
            return self._cached_model_name
        try:
            self._cached_model_name = self.embedding_handler.get_model_dict()["model"]
        except (KeyError, TypeError):
            # Fallback: ask the provider and record its own model id.
            response, _ = self.embedding_handler.get_embedding_and_cost(["_probe_"])
            self._cached_model_name = getattr(response, "model", "unknown")
        return self._cached_model_name

    def get_embed_dimension(self) -> int:
        if self._cached_embed_dimension is None:
            vectors = self.get_embeddings(["_probe_"])
            self._cached_embed_dimension = len(vectors[0])
        return self._cached_embed_dimension

    def get_usage_counter(self) -> UsageCounter:
        return self.usage_counter

    def reset_usage_counter(self) -> None:
        self.usage_counter.reset_counts()


# ---------------------------------------------------------------------------
# LLM side
# ---------------------------------------------------------------------------

class LlmHandler(ABC):
    """Interface a user-written LLM adapter must satisfy.

    Handlers accept OpenAI-style ``messages`` and return LiteLLM's
    ``ModelResponse`` (which mirrors the OpenAI response shape), plus the
    USD cost of the call.
    """

    @abstractmethod
    def get_completion_and_cost(
        self, messages: list[dict]
    ) -> Tuple[ModelResponse, float]: ...

    @abstractmethod
    async def aget_completion_and_cost(
        self, messages: list[dict]
    ) -> Tuple[ModelResponse, float]: ...

    @abstractmethod
    def get_model_dict(self) -> dict: ...

    @abstractmethod
    def get_max_concurrent_calls(self) -> int: ...


class Llm:
    """Package-side LLM wrapper.

    Wraps a user-supplied ``LlmHandler`` with usage accounting and a
    semaphore. Exposes ``bound_run_llm(prompt)`` to match the call shape
    the pipeline already uses: a single-user-message prompt in, a dict
    ``{"content", "logprobs", "duration", "cost"}`` out.
    """

    def __init__(self, llm_handler: LlmHandler,
                 usage_counter: Optional[UsageCounter] = None,
                 print_query_duration: bool = False):
        self.llm_handler = llm_handler
        self.usage_counter = usage_counter if usage_counter is not None else UsageCounter()
        self.llm_semaphore = asyncio.Semaphore(llm_handler.get_max_concurrent_calls())
        self.print_query_duration = print_query_duration
        self._cached_model_name: Optional[str] = None

    def __repr__(self) -> str:
        return f"Llm(model_name={self.get_model_name()!r})"

    @property
    def token_counter(self) -> UsageCounter:
        return self.usage_counter

    @property
    def model_name(self) -> str:
        return self.get_model_name()

    def get_model_name(self) -> str:
        if self._cached_model_name is None:
            try:
                self._cached_model_name = self.llm_handler.get_model_dict()["model"]
            except (KeyError, TypeError):
                self._cached_model_name = "unknown"
        return self._cached_model_name

    def get_usage_counter(self) -> UsageCounter:
        return self.usage_counter

    def reset_usage_counter(self) -> None:
        self.usage_counter.reset_counts()

    def create_llm_costs_dict(self) -> dict:
        """Return a dict of usage/cost metrics for logs.json / MLflow."""
        usage = self.usage_counter.get_usage_dict()
        return {
            "embedding_tokens": usage["embedding_tokens"],
            "llm_prompt_tokens": usage["llm_prompt_tokens"],
            "llm_completion_tokens": usage["llm_completion_tokens"],
            "total_llm_token_count": usage["total_llm_token_count"],
            "total_llm_costs_in_euro": usage["total_cost_in_dollar"],
        }

    async def bound_run_llm(self, formatted_prompt: str) -> Tuple[dict, Optional[BaseException]]:
        """Run the LLM under the concurrency semaphore.

        Returns ``(response_dict, error)`` where ``response_dict`` has
        ``content``, ``logprobs``, ``duration``, ``cost``. On error,
        ``response_dict`` has empty ``content`` and ``logprobs=None``.
        """
        async with self.llm_semaphore:
            return await self._run_llm(formatted_prompt)

    async def _run_llm(self, formatted_prompt: str) -> Tuple[dict, Optional[BaseException]]:
        messages = [{"role": "user", "content": formatted_prompt}]
        start = time.perf_counter()
        try:
            response, cost = await self.llm_handler.aget_completion_and_cost(messages)
        except Exception as e:  # broad by design — handlers raise provider-specific errors
            logger.warning("LLM call failed: %s", e)
            return {"content": "", "logprobs": None, "duration": 0.0, "cost": 0.0}, e

        duration = time.perf_counter() - start
        if self.print_query_duration:
            logger.info("LLM query duration: %.2f seconds", duration)

        content, logprobs = self._extract_content_and_logprobs(response)
        self._record_usage(response, cost)

        return {
            "content": content,
            "logprobs": logprobs,
            "duration": duration,
            "cost": float(cost or 0.0),
        }, None

    @staticmethod
    def _extract_content_and_logprobs(response: ModelResponse) -> Tuple[str, Any]:
        try:
            choice = response.choices[0]
        except (AttributeError, IndexError):
            return "", None

        message = getattr(choice, "message", None)
        if message is None and isinstance(choice, dict):
            message = choice.get("message")
        content = getattr(message, "content", None) if message is not None else None
        if content is None and isinstance(message, dict):
            content = message.get("content")
        logprobs = getattr(choice, "logprobs", None)
        if logprobs is None and isinstance(choice, dict):
            logprobs = choice.get("logprobs")
        return content or "", logprobs

    def _record_usage(self, response: ModelResponse, cost: float) -> None:
        try:
            usage = response.usage
            self.usage_counter.add_prompt_tokens(int(getattr(usage, "prompt_tokens", 0)))
            self.usage_counter.add_completion_tokens(int(getattr(usage, "completion_tokens", 0)))
        except AttributeError:
            pass
        self.usage_counter.add_cost(cost)
