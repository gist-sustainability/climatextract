# Custom Providers

climatextract ships with reference Azure AI Foundry and Azure OpenAI Service adapters. To use a different LLM or embedding provider — OpenAI direct, Anthropic, a local model, anything else — implement a handler and pass it to `extract()`.

---

## When you need this

Reach for a custom handler when:

- You want to use a provider the bundled Azure adapters don't cover.
- You want to use Azure but need behavior the bundled adapters don't expose.
- You're running models locally and don't want to go through a cloud provider at all.

If you just want to tweak parameters on an Azure model (temperature, reasoning effort, concurrency), you can instantiate `AzureAIFoundryLlmHandler` or `AzureOpenAILlmHandler` directly without subclassing — see [Running Extraction](running-extraction.md#using-a-custom-llm-or-embedding-provider).

---

## The interfaces

Two abstract base classes in `climatextract.llm_embedding_api_bridge` define the contract.

### `LlmHandler`

Subclasses must implement:

| Method | Purpose |
|--------|---------|
| `get_completion_and_cost(messages)` | Sync call. Takes OpenAI-style `messages` list, returns `(response, cost_in_usd)`. |
| `aget_completion_and_cost(messages)` | Async version of the above. |
| `get_model_dict()` | Returns a dict describing the model (must include a `"model"` key — used for logging and `repr`). |
| `get_max_concurrent_calls()` | Returns the max number of in-flight calls the wrapper's semaphore should allow. |

The returned `response` should be shaped like an OpenAI chat completion — specifically, `response.choices[0].message.content` (for the output text), `response.choices[0].logprobs` (optional), and `response.usage.prompt_tokens` / `response.usage.completion_tokens`. LiteLLM's `ModelResponse` already matches this shape, as do OpenAI SDK responses.

### `EmbeddingModelHandler`

Subclasses must implement:

| Method | Purpose |
|--------|---------|
| `get_embedding_and_cost(texts)` | Sync call. Takes `list[str]`, returns `(response, cost_in_usd)`. |
| `aget_embedding_and_cost(texts)` | Async version. |
| `get_model_dict()` | Dict with a `"model"` key. |
| `get_max_concurrent_calls()` | Max in-flight embedding calls. |

The returned `response` should expose `response.data` as a sequence of items with an `embedding` field (either attribute- or dict-style — both are handled), and `response.usage.prompt_tokens`.

---

## Minimal example: OpenAI direct

An LLM handler that uses the OpenAI SDK directly, no LiteLLM:

```python
import os
from typing import Tuple
from openai import AsyncOpenAI, OpenAI

from climatextract.llm_embedding_api_bridge import LlmHandler


class OpenAILlmHandler(LlmHandler):
    def __init__(self, model: str = "gpt-4o-mini", max_concurrent_calls: int = 8):
        self.model = model
        self._max = max_concurrent_calls
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._aclient = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def get_model_dict(self) -> dict:
        return {"model": self.model}

    def get_max_concurrent_calls(self) -> int:
        return self._max

    def get_completion_and_cost(self, messages: list[dict]) -> Tuple[object, float]:
        response = self._client.chat.completions.create(model=self.model, messages=messages)
        return response, 0.0  # compute your own cost if you need it

    async def aget_completion_and_cost(self, messages: list[dict]) -> Tuple[object, float]:
        response = await self._aclient.chat.completions.create(model=self.model, messages=messages)
        return response, 0.0
```

Pass it to `extract()`:

```python
from climatextract import extract

result_path = extract("./data/pdfs/", llm=OpenAILlmHandler(model="gpt-4o-mini"))
```

---

## Notes

- **Don't manage concurrency inside your handler.** The `Llm` / `EmbeddingModel` wrapper already applies a semaphore sized by `get_max_concurrent_calls()`.
- **Don't count tokens or cost inside your handler beyond returning them.** The wrapper records them via `UsageCounter` and surfaces them in `logs.json` and MLflow.
- **Cost is optional.** Return `0.0` if your provider doesn't give you a price back. The pipeline will still run.
- **Reference implementations.** For more complete examples — routing, Azure AD token handling, per-model parameter quirks — see [`climatextract/adapters/azure_ai_foundry.py`](https://github.com/gist-sustainability/climatextract/blob/main/climatextract/adapters/azure_ai_foundry.py) and [`climatextract/adapters/azure_openai.py`](https://github.com/gist-sustainability/climatextract/blob/main/climatextract/adapters/azure_openai.py).
- **Subclass instead of write from scratch.** If you only need to tweak one or two things (e.g., an extra request param), subclassing `AzureAIFoundryLlmHandler` or `AzureOpenAILlmHandler` and overriding `__init__` is usually enough.
