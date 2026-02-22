"""Key idea:
LiteLLM provides access to a wide range of LLMs and embedding models. Users specify the model they want in LiteLLM syntax. We don't need to accomodate new models, meaning less hassle for us.
Importantly, it is the users responsibility keep litellm (and all the most recent models) up to date. We don't import it, meaning that no updates are needed from our side.

All API calls are returned in a standardized OpenAI output format.
Moreover, LiteLLM helps calculating cost and token usage."""

from abc import ABC, abstractmethod
import asyncio
import threading
import time
from typing import Tuple

from litellm import EmbeddingResponse

class UsageCounter:
    """
    Counts tokens for LLM calls and embeddings.
    """

    def __init__(self):
        self.reset_counts()

    def reset_counts(self):
        """Reset all counts to zero."""
        self.prompt_llm_token_count = 0
        self.completion_llm_token_count = 0
        self.total_llm_token_count = 0
        self.total_embedding_token_count = 0
        self.total_cost = 0

    def get_usage_dict(self):
        """Creates a dictionary with the costs of the LLM."""
        usage_dict = {
            "embedding_tokens": self.total_embedding_token_count,
            "llm_prompt_tokens": self.prompt_llm_token_count,
            "llm_completion_tokens": self.completion_llm_token_count,
            "total_llm_token_count": self.total_llm_token_count,
            "total_cost_in_dollar": self.total_cost,
        }
        return usage_dict

    def add_prompt_tokens(self, count):
        """Add tokens to the prompt token count."""
        self.prompt_llm_token_count += count

    def add_completion_tokens(self, count):
        """Add tokens to the completion token count."""
        self.completion_llm_token_count += count

    def add_total_tokens(self, count):
        """Add tokens to the total token count."""
        self.total_llm_token_count += count

    def add_cost(self, cost):
        """Add cost to the total cost."""
        self.total_cost += cost

    def add_embedding_tokens(self, count):
        """Add tokens to the embedding token count."""
        self.total_embedding_token_count += count


class ThreadSafeTokenProvider: 
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


class EmbeddingModelHandler(ABC):

    @abstractmethod
    def get_embedding_and_cost(self, texts: list[str]) -> Tuple[EmbeddingResponse, float]: ...

    @abstractmethod
    async def aget_embedding_and_cost(self, texts: list[str]) -> Tuple[EmbeddingResponse, float]: ...

    @abstractmethod
    def get_model_dict(self) -> dict: ...

    @abstractmethod
    def get_max_concurrent_calls(self) -> int: ...


class EmbeddingModel:
    def __init__(self, embedding_handler: EmbeddingModelHandler, 
                 usage_counter: UsageCounter = UsageCounter()):

        self.embedding_handler = embedding_handler
        self.usage_counter = usage_counter
        self.embedding_semaphore = asyncio.Semaphore(embedding_handler.get_max_concurrent_calls())


    ## TODO, __repr__ is not how we can identify the model.
    def __repr__(self):
        return f"{self.__class__.__name__}(model_name={self.MODEL!r}, api_version={self.API_VERSION!r})"
    
    def get_embedding(self, texts: list[str]) -> list[dict[float]]:
        """Create embeddings for a list of texts."""

        response, costs = self.embedding_handler.get_embedding_and_cost(texts)

        self.usage_counter.add_embedding_tokens(response.usage.prompt_tokens)
        self.usage_counter.add_cost(costs)

        return response.data
    
    async def bounded_aget_embedding(self, texts: list[str]) -> list[dict[float]]:
        """Create embeddings for a list of texts asynchronously with semaphore control."""

        async with self.embedding_semaphore:
            return await self.aget_embedding(texts)

    async def aget_embedding(self, texts: list[str]) -> list[dict[float]]:
        """Create embeddings for a list of texts asynchronously."""

        response, costs = self.embedding_handler.aget_embedding_and_cost(texts)

        self.usage_counter.add_embedding_tokens(response.usage.prompt_tokens)
        self.usage_counter.add_cost(costs)

        return response.data
    
    def get_model_name(self) -> str:
        try:
            modelname = self.embedding_handler.get_model_dict()["model"]
        except KeyError:
            modelname = self.embedding_handler.get_embedding(["run embed model"])[0].model
        return modelname

    def get_embed_dimension(self) -> int:
        return len(self.get_embedding(["get embed dimension"])[0]['embedding'])
    
    def get_usage_counter(self) -> UsageCounter:
        return self.usage_counter
    
    def reset_usage_counter(self):
        self.usage_counter.reset_counts()