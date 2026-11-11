import asyncio
from types import SimpleNamespace

import tiktoken

from climatextract import llm_embedding_api_bridge as bridge


class FakeEmbeddingHandler:
    def __init__(self):
        self.calls = []
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def get_max_concurrent_calls(self):
        return 1

    def get_model_dict(self):
        return {"model": "fake"}

    def _response(self, texts):
        self.calls.append(texts)
        token_count = sum(len(self.encoding.encode(text)) for text in texts)
        data = [SimpleNamespace(embedding=[float(len(text))]) for text in texts]
        response = SimpleNamespace(
            data=data,
            usage=SimpleNamespace(prompt_tokens=token_count),
        )
        return response, float(len(texts))

    def get_embedding_and_cost(self, texts):
        return self._response(texts)

    async def aget_embedding_and_cost(self, texts):
        return self._response(texts)


def _assert_batched_results(model, handler, texts, expected_tokens):
    assert model.get_embeddings(texts) == [[float(len(text))] for text in texts]
    assert [text for call in handler.calls for text in call] == texts
    assert all(
        sum(len(handler.encoding.encode(text)) for text in call)
        <= expected_tokens
        for call in handler.calls
    )
    assert len(handler.calls) > 1
    assert model.usage_counter.total_embedding_token_count == sum(
        len(handler.encoding.encode(text)) for text in texts
    )
    assert model.usage_counter.total_cost == float(len(texts))


def test_sync_embeddings_are_split_into_bounded_batches(monkeypatch):
    texts = ["red orange yellow", "green blue purple", "black white gray"]
    encoding = tiktoken.get_encoding("cl100k_base")
    max_tokens = max(len(encoding.encode(text)) for text in texts)
    monkeypatch.setattr(bridge, "_MAX_EMBEDDING_BATCH_TOKENS", max_tokens)

    handler = FakeEmbeddingHandler()
    model = bridge.EmbeddingModel(handler)

    _assert_batched_results(model, handler, texts, max_tokens)


def test_async_embeddings_are_split_into_bounded_batches(monkeypatch):
    texts = ["red orange yellow", "green blue purple", "black white gray"]
    encoding = tiktoken.get_encoding("cl100k_base")
    max_tokens = max(len(encoding.encode(text)) for text in texts)
    monkeypatch.setattr(bridge, "_MAX_EMBEDDING_BATCH_TOKENS", max_tokens)

    handler = FakeEmbeddingHandler()
    model = bridge.EmbeddingModel(handler)

    vectors = asyncio.run(model.aget_embeddings(texts))

    assert vectors == [[float(len(text))] for text in texts]
    assert [text for call in handler.calls for text in call] == texts
    assert all(
        sum(len(handler.encoding.encode(text)) for text in call)
        <= max_tokens
        for call in handler.calls
    )
    assert len(handler.calls) > 1
    assert model.usage_counter.total_embedding_token_count == sum(
        len(handler.encoding.encode(text)) for text in texts
    )
    assert model.usage_counter.total_cost == float(len(texts))
