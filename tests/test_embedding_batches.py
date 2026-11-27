import asyncio
from types import SimpleNamespace

import pytest

from climatextract import llm_embedding_api_bridge as bridge


class FakeEmbeddingHandler:
    """Returns one vector per text, [len(text)], and records every request."""

    def __init__(self, model="azure/text-embedding-3-large"):
        self.model = model
        self.calls = []

    def get_max_concurrent_calls(self):
        return 1

    def get_model_dict(self):
        return {"model": self.model}

    def get_embedding_and_cost(self, texts):
        self.calls.append(texts)
        data = [SimpleNamespace(embedding=[float(len(text))]) for text in texts]
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=len(texts))), 1.0

    async def aget_embedding_and_cost(self, texts):
        return self.get_embedding_and_cost(texts)


TEXTS = ["red orange yellow", "green blue purple", "black white gray"]


def test_encoding_is_chosen_from_model_name(monkeypatch):
    # Use a fallback that differs from the real encoding (cl100k_base), so the
    # test can tell a model lookup from a fallback.
    monkeypatch.setattr(bridge, "_FALLBACK_ENCODING", "r50k_base")
    model = bridge.EmbeddingModel(FakeEmbeddingHandler("azure/text-embedding-3-large"))
    assert model.count_tokens("red orange yellow") == 3
    assert model._encoding.name == "cl100k_base"


def test_unknown_model_falls_back_to_default_encoding(monkeypatch):
    monkeypatch.setattr(bridge, "_FALLBACK_ENCODING", "r50k_base")
    model = bridge.EmbeddingModel(
        FakeEmbeddingHandler("sentence-transformers/all-MiniLM-L6-v2"))
    model.count_tokens("red orange yellow")
    assert model._encoding.name == "r50k_base"


def test_truncate_cuts_long_text_and_keeps_short_text():
    model = bridge.EmbeddingModel(FakeEmbeddingHandler())
    text = "red orange yellow green"  # 4 tokens
    assert model.truncate(text, max_tokens=4) == text
    assert model.truncate(text, max_tokens=2) == "red orange"
    long_text = " ".join([text] * 5000)  # 20,000 tokens
    assert model.count_tokens(long_text) > bridge._MAX_TOKENS_PER_TEXT
    # Re-encoding a decoded prefix may merge tokens differently, so only
    # require that the result fits.
    assert model.count_tokens(model.truncate(long_text)) <= bridge._MAX_TOKENS_PER_TEXT


def test_text_over_per_text_limit_raises(monkeypatch):
    monkeypatch.setattr(bridge, "_MAX_TOKENS_PER_TEXT", 2)
    handler = FakeEmbeddingHandler()
    with pytest.raises(ValueError, match="maximum is 2 tokens per text"):
        bridge.EmbeddingModel(handler).get_embeddings(TEXTS)
    assert handler.calls == []


@pytest.mark.parametrize("use_async", [False, True])
def test_embeddings_are_split_by_token_limit(monkeypatch, use_async):
    # Each text has 3 tokens, so at most 2 texts fit into a 6-token request.
    monkeypatch.setattr(bridge, "_MAX_TOKENS_PER_REQUEST", 6)
    handler = FakeEmbeddingHandler()
    model = bridge.EmbeddingModel(handler)

    if use_async:
        vectors = asyncio.run(model.aget_embeddings(TEXTS))
    else:
        vectors = model.get_embeddings(TEXTS)

    assert handler.calls == [TEXTS[:2], TEXTS[2:]]
    assert vectors == [[float(len(text))] for text in TEXTS]
    # The fake handler reports one "token" per text and costs 1.0 per request,
    # so these sums show that usage from both requests (2 + 1 texts) is recorded.
    assert model.usage_counter.total_embedding_token_count == 3
    assert model.usage_counter.total_cost == 2.0


def test_embeddings_are_split_by_number_of_texts(monkeypatch):
    monkeypatch.setattr(bridge, "_MAX_TEXTS_PER_REQUEST", 2)
    handler = FakeEmbeddingHandler()
    model = bridge.EmbeddingModel(handler)

    model.get_embeddings(TEXTS)

    assert handler.calls == [TEXTS[:2], TEXTS[2:]]


def test_small_input_is_sent_in_one_request():
    handler = FakeEmbeddingHandler()
    bridge.EmbeddingModel(handler).get_embeddings(TEXTS)
    assert handler.calls == [TEXTS]


@pytest.mark.parametrize("use_async", [False, True])
def test_empty_list_raises_without_request(use_async):
    handler = FakeEmbeddingHandler()
    model = bridge.EmbeddingModel(handler)
    with pytest.raises(ValueError, match="empty list"):
        if use_async:
            asyncio.run(model.aget_embeddings([]))
        else:
            model.get_embeddings([])
    assert handler.calls == []


@pytest.mark.parametrize("texts", [[""], ["red", "", "blue"]])
def test_empty_strings_are_passed_to_provider_unchanged(texts):
    # Empty strings are not handled locally: the provider receives them and
    # decides (Azure rejects them with HTTP 400).
    handler = FakeEmbeddingHandler()
    bridge.EmbeddingModel(handler).get_embeddings(texts)
    assert handler.calls == [texts]
