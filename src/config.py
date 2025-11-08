import asyncio
import os
import time
import datetime
import threading 
from typing import List

import tiktoken
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

# from llama_index.core.callbacks import TokenCountingHandler
# from llama_index.llms.azure_openai import AzureOpenAI
from openai import APIStatusError, AsyncAzureOpenAI, RateLimitError

load_dotenv()  # load environment variables from .env file

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
                    print("Token renewed, valid until:", 
                        datetime.datetime.fromtimestamp(self.expires_on).strftime('%A, %B %d, %Y %H:%M:%S')) 
        return self.token 

try:
    from azure_authentication import customized_azure_login
    credential = customized_azure_login.CredentialFactory().select_credential()
except Exception as e:
    print(f"Error applying azure authentication, using API key: {e}")
    credential = None  


# Custom token counter to replace TokenCountingHandler
class TokenCounter:
    """
    Custom token counter to replace llama_index.core.callbacks.TokenCountingHandler.
    Counts tokens for LLM calls and embeddings.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.reset_counts()

    def reset_counts(self):
        """Reset all token counts to zero."""
        self.prompt_llm_token_count = 0
        self.completion_llm_token_count = 0
        self.total_llm_token_count = 0
        self.total_embedding_token_count = 0

    def count_tokens(self, text):
        """Count tokens in a text string using the provided tokenizer."""
        if not isinstance(text, str):
            # Handle non-string inputs by converting to string or returning 0
            if text is None:
                return 0
            try:
                text = str(text)
            except:
                return 0

        return len(self.tokenizer(text))

    def add_prompt_tokens(self, count):
        """Add tokens to the prompt token count."""
        self.prompt_llm_token_count += count
        self.total_llm_token_count += count

    def add_completion_tokens(self, count):
        """Add tokens to the completion token count."""
        self.completion_llm_token_count += count
        self.total_llm_token_count += count

    def add_embedding_tokens(self, count):
        """Add tokens to the embedding token count."""
        self.total_embedding_token_count += count

    def update_from_response(self, response):
        """Update token counts from an OpenAI API response."""
        if hasattr(response, "usage"):
            self.add_prompt_tokens(response.usage.prompt_tokens)
            self.add_completion_tokens(response.usage.completion_tokens)
            return True
        return False


class Llm:
    """

    Contains LLM model

    """

    def __init__(
        self,
        model_name="gpt-4o-mini-2024-07-18",
        api_version=os.environ["API_VERSION"],
        return_logprobs: bool = False
    ):

        self.model_name = model_name
        self.azure_deployment = model_name
        self.api_version = api_version
        self.return_logprobs = return_logprobs
        self.azure_endpoint = os.environ["AZURE_ENDPOINT"]

        self.api_key = os.getenv("API_KEY")
        if not self.api_key:
            login_token_provider = ThreadSafeTokenProvider(credential)
            self.api_key = login_token_provider()

        bbk_models = [
            "o3",
            "o1-preview",
            "o1-mini",
            "gpt-4.1",
            "gpt-4o-se",
            "gpt-4o-mini-se",
            "gpt-4-turbo-se",
            "gpt-4-se",
            "gpt-35-turbo-se",
        ]
        # Set configuration exceptions based on model
        if self.model_name == "gpt-35-turbo-16k":
            self.azure_deployment = "gpt-35-turbo-0301"
            self.azure_endpoint = os.environ["AZURE_ENDPOINT_GIST_PROJECT_WESTEUROPE"]
            self.max_parallel_llm_prompts_running = 8
            self.token_counter = TokenCounter(
                tokenizer=tiktoken.encoding_for_model("gpt-35-turbo").encode
            )

        elif self.model_name == "gpt-4-1106-preview":
            raise Exception(
                "Model gpt-4-1106-preview does not exist anymore. It was deactivated by Azure, please use gpt-4o-2024-11-20 instead."
            )

        elif (
            self.model_name == "gpt-4o-mini-2024-07-18"
            or self.model_name == "gpt-4o-2024-11-20"
            or self.model_name in bbk_models
        ):
            self.max_parallel_llm_prompts_running = 25
                    # Initialize token counter
            # both gpt-3.5-turbo and gpt-4 are based on the same cl100k_base encoding -> doesn't matter which model we use here
            # gpt-4o, o1, and o3 all use the same o200k_base encoding -> doesn't matter which model we use here
            self.token_counter = TokenCounter(
            tokenizer=tiktoken.encoding_for_model(
                "gpt-4o" if "4o" in self.model_name or "o3" in self.model_name else "gpt-4"
            ).encode
        )

        elif self.model_name == "o3-mini-2025-01-31":
            self.api_version = "2024-12-01-preview"
            self.max_parallel_llm_prompts_running = 2
            self.token_counter = TokenCounter(
                tokenizer=tiktoken.encoding_for_model("o3").encode
            )

        elif self.model_name == "gpt-oss-120b":
            # Modell-ID: azureml://registries/azureml-openai-oss/models/gpt-oss-120b/versions/4
            self.api_version = "2025-01-01-preview"
            self.azure_endpoint = os.environ["AZURE_AI_FOUNDRY_ENDPOINT"]
            self.max_parallel_llm_prompts_running = 25
            self.token_counter = TokenCounter(
                tokenizer=tiktoken.encoding_for_model("gpt-oss-120b").encode
            )

        elif self.model_name == "gpt-4.1-2025-04-14":
            # Modell-ID: azureml://registries/azureml-openai-oss/models/gpt-oss-120b/versions/4
            self.api_version = "2025-01-01-preview"
            self.azure_endpoint = os.environ["AZURE_AI_FOUNDRY_ENDPOINT"]
            self.max_parallel_llm_prompts_running = 4
            self.token_counter = TokenCounter(
                tokenizer=tiktoken.encoding_for_model("gpt-4.1").encode
            )

        elif self.model_name == "gpt-5-chat-2025-08-07":
            # Modell-ID: azureml://registries/azureml-openai-oss/models/gpt-oss-120b/versions/4
            self.api_version = "2025-01-01-preview"
            self.azure_endpoint = os.environ["AZURE_AI_FOUNDRY_ENDPOINT"]
            self.max_parallel_llm_prompts_running = 8
            self.token_counter = TokenCounter(
                tokenizer=tiktoken.encoding_for_model("gpt-5-chat").encode
            )
       

        elif self.model_name == "o1-2024-1217":
            raise Exception("Model not implemented yet")

        else:
            raise Exception(f"Unknown model name: {self.model_name}")

        # Initialize the OpenAI client directly
        # Argumente für den Client vorbereiten
        client_args = {
            "azure_endpoint": self.azure_endpoint,
            "api_version": self.api_version,
            "max_retries": 4,
            "timeout": 340.0,
        }
        if self.api_key:
            client_args["api_key"] = self.api_key
        else:
            client_args["azure_ad_token_provider"] = (
                login_token_provider  # wird automatisch aktualisiert, falls die Klasse das unterstützt
            )

        # Client initialisieren
        self.client = AsyncAzureOpenAI(**client_args)

        self.semaphore = asyncio.Semaphore(self.max_parallel_llm_prompts_running)

    def __repr__(self):
        return f"{self.__class__.__name__}(model_name={self.model_name!r}, api_version={self.api_version!r})"

    def calculate_llm_calling_price(self, input_tokens, output_tokens):
        """
        Cost calculator
        based on prices from https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/#pricing
        or from the model catalogue at https://ai.azure.com/
        """

        if self.model_name == "gpt-35-turbo-16k":
            return 0.4759 * input_tokens / 1000000 + 1.4277 * output_tokens / 1000000
        # elif self.model_name == "gpt-4-1106-preview":
        #    return 9.518 * input_tokens / 1000000 + 28.554 * output_tokens / 1000000
        elif self.model_name == "gpt-4o-2024-11-20":
            return 2.16967 * input_tokens / 1000000 + 8.6787 * output_tokens / 1000000
        elif self.model_name == "gpt-4o-mini-2024-07-18":
            return 0.1319 * input_tokens / 1000000 + 0.5208 * output_tokens / 1000000
        elif self.model_name == "o3-mini-2025-01-31":
            return 0.9547 * input_tokens / 1000000 + 3.818616 * output_tokens / 1000000
        elif self.model_name == "o1-2024-1217":
            return 13.0181 * input_tokens / 1000000 + 52.072033 * output_tokens / 1000000
        elif self.model_name == "gpt-oss-120b":
            return 0.3 * input_tokens / 1000000 + 2.5 * output_tokens / 1000000
        elif self.model_name == "gpt-4.1-2025-04-14":
            return 2 * input_tokens / 1000000 + 8 * output_tokens / 1000000
        elif self.model_name == "gpt-5-chat-2025-08-07":
            return 1.25 * input_tokens / 1000000 + 10 * output_tokens / 1000000
        elif self.model_name == "Llama-4-Maverick-17B-128E-Instruct-FP8":
            return 0.35 * input_tokens / 1000000 + 1.41 * output_tokens / 1000000 # according to this site it is a bit cheaper: https://azure.microsoft.com/en-us/pricing/details/phi-3/#pricing
        
        
        else:
            return -1.0

    def create_llm_costs_dict(self):
        """Creates a dictionary with the costs of the LLM."""
        llm_costs = {
            "embedding_tokens": self.token_counter.total_embedding_token_count,
            "llm_prompt_tokens": self.token_counter.prompt_llm_token_count,
            "llm_completion_tokens": self.token_counter.completion_llm_token_count,
            "total_llm_token_count": self.token_counter.total_llm_token_count,
            "total_llm_costs_in_euro": self.calculate_llm_calling_price(
                self.token_counter.prompt_llm_token_count,
                self.token_counter.completion_llm_token_count,
            ),
        }

        return llm_costs

    async def bound_run_llm(self, formatted_prompt):
        async with self.semaphore:
            return await self.run_llm(formatted_prompt=formatted_prompt)

    async def run_llm(self, formatted_prompt, print_query_duration=True):
        cur_time = time.perf_counter()

        try:
            # Correctly pass a boolean for the logprobs parameter
            response = await self.client.chat.completions.create(
                model=self.azure_deployment,
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=0.0,
                logprobs=self.return_logprobs
            )

            # Use the QueryPipeline without CallbackManager
            # results = await p.arun_multi({"llm_prompt": {"context_str": doc_text}})
            # raw_response = results["llm"]["output"]

            raw_response = response.choices[0].message.content
            logprob_data = response.choices[0].logprobs if self.return_logprobs else None
            
            # First try to get token counts from response usage (most accurate)
            token_usage_found = False
            if hasattr(self.token_counter, "update_from_response"):
                try:
                    token_usage_found = self.token_counter.update_from_response(response)
                except Exception as e:
                    print(f"Error updating token counts from response: {e}")

            # Only count tokens manually if we couldn't get them from usage
            if not token_usage_found:
                if hasattr(self.token_counter, "count_tokens"):
                    try:
                        # Count prompt tokens
                        prompt_tokens = self.token_counter.count_tokens(formatted_prompt)
                        self.token_counter.add_prompt_tokens(prompt_tokens)

                        # Count completion tokens
                        if raw_response:
                            completion_tokens = self.token_counter.count_tokens(raw_response)
                            self.token_counter.add_completion_tokens(completion_tokens)
                    except Exception as e:
                        print(f"Error counting tokens: {e}")

            if print_query_duration:
                duration_time = time.perf_counter() - cur_time
                print("LLM query execution time: " + str(duration_time) + " seconds")

            return {"content": raw_response, "logprobs": logprob_data}, None

        except RateLimitError as e:
            print("A 429 status code (Rate Limit error) was received; we should back off a bit.")

            raw_response = ""

            return {"content": raw_response, "logprobs": None}, e

        except APIStatusError as e:
            raw_response = ""

            print("Another non-200-range status code was received")
            print(e.status_code)
            print(e.response)

            return {"content": raw_response, "logprobs": None}, e

        except RuntimeError as e:
            raw_response = ""
            print("RuntimeError: ", e)

            return {"content": raw_response, "logprobs": None}, e


    def print_llm_costs(self):
        """Prints the costs of the LLM."""
        print(
            "Embedding Tokens (used for search, but not counted here): ",
            self.token_counter.total_embedding_token_count,
            "\n",
            "LLM Prompt Tokens: ",
            self.token_counter.prompt_llm_token_count,
            "\n",
            "LLM Completion Tokens: ",
            self.token_counter.completion_llm_token_count,
            "\n",
            "Total LLM Token Count: ",
            self.token_counter.total_llm_token_count,
            "\n",
            "Total LLM costs (Euro): ",
            self.calculate_llm_calling_price(
                self.token_counter.prompt_llm_token_count,
                self.token_counter.completion_llm_token_count,
            ),
            "\n",
            "We now reset the token counter to zero.",
            "\n",
        )


class EmbeddingModel:
    # TODO We should change to text-embedding-3-small when it becomes available

    # Use AzureOpenAI Embeddings with llama.index

    # Alternative: Use HuggingFace Embeddings locally(!!) with llama.index
    # Requires:
    # from llama_index.embeddings.huggingface import HuggingFaceEmbedding (more than 1.5GB disk space needed)
    # embed_model = HuggingFaceEmbedding(
    #     model_name="BAAI/bge-small-en-v1.5"
    # )
    """
    Class that contains embedding model
    """

    def __init__(self, model_name="text-embedding-ada-002", api_version=os.environ["API_VERSION"]):
        # or text-embedding-3-large

        self.model_name = model_name
        self.api_version = api_version

        if "sentence-transformers" in model_name:
            # Don't make these dependencies a requirement because HuggingFace models will probably not used that often
            import torch
            from sentence_transformers import SentenceTransformer

            # HuggingFace backend
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.backend = "hf"
            self.model = SentenceTransformer(model_name, device=device)
            self.embed_dimension = self.model.get_sentence_embedding_dimension()
            self.embedding_semaphore = asyncio.Semaphore(1)
            self.embed_batch_size = 64

            print(
                f"Warning: Could not initialize tokenizer for EmbeddingModel {model_name}. Counting words instead."
            )
            self.token_counter = TokenCounter(tokenizer=lambda text: text.split())

        else:
            # Azure/OpenAI backend
            self.backend = "azure"
            concurrent_calls = 1  # values other than 1 may not work properly because AzureOpenAIEmbedding does not support asynchronous calls
            self.embedding_semaphore = asyncio.Semaphore(concurrent_calls)

            if self.model_name == "text-embedding-ada-002":
                self.embed_dimension = 1536
            elif self.model_name == "text-embedding-3-large":
                self.embed_dimension = 3072
            else:
                raise Exception("Unknown model name")

            self.azure_endpoint = os.environ["AZURE_ENDPOINT"]
            self._initiate_embedding_client()

            self.token_counter = TokenCounter(
                tokenizer=tiktoken.encoding_for_model(model_name).encode
            )

    def __repr__(self):
        return f"{self.__class__.__name__}(model_name={self.model_name!r}, api_version={self.api_version!r})"

    def _initiate_embedding_client(self):
        if self.backend == "azure":
            self.api_key = os.getenv("API_KEY")
            if not self.api_key:
                login_token_provider = ThreadSafeTokenProvider(credential)
                self.api_key = login_token_provider()
            self.deployment_name = os.getenv("DEPLOYMENT_NAME_TEXT_EMBEDDING_3_LARGE")
            if not self.deployment_name:
                self.deployment_name = self.model_name

            self.embed_model = AzureOpenAIEmbedding(
                model=self.model_name,
                embed_batch_size=250,
                deployment_name=self.deployment_name,
                azure_endpoint=self.azure_endpoint,
                # use_azure_ad=True, # only useful for debugging purposes?
                api_version=self.api_version,
                api_key=self.api_key,
            )

            Settings.embed_model = self.embed_model

        # get_token("https://cognitiveservices.azure.com/.default").expires_on == 1741262950

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        texts_with_content = ["empty parsed page" if item == "" else item for item in texts]

        if self.backend == "hf":
            if hasattr(self, "token_counter"):
                for text in texts_with_content:
                    tokens = self.token_counter.count_tokens(text)
                    self.token_counter.add_embedding_tokens(tokens)
            return self.model.encode(
                texts_with_content, batch_size=self.embed_batch_size, show_progress_bar=True
            ).tolist()

        else:
            # certainly not optimal: we re-initiate the client with every call, but one would only need to update the token every 1h.
            self._initiate_embedding_client()

            # return self.embed_model.get_text_embedding_batch(texts_with_content)
            embeddings = self.embed_model.get_text_embedding_batch(texts_with_content)
            # return self.embed_model._get_text_embeddings(texts_with_content) # this won't work for very long vectors: Error 429 Too Many Requests

            if hasattr(self, "token_counter"):
                total_tokens_for_batch = 0
                for text in texts_with_content:
                    tokens = self.token_counter.count_tokens(text)
                    total_tokens_for_batch += tokens
                self.token_counter.add_embedding_tokens(total_tokens_for_batch)

            return embeddings

    # async def aget_embeddings(self, texts: List[str]) -> List[List[float]]:
    # throws AttributeError: 'AzureOpenAIEmbedding' object has no attribute '_async_http_client'
    # because AzureOpenAIEmbedding in package llama.index does not support asynchronous calls.

    #     texts_with_content = [
    #         "empty parsed page" if item == "" else item for item in texts]

    #     async with self.embedding_semaphore:
    #         embeddings = await self.embed_model._aget_text_embeddings(texts_with_content)

    #     return embeddings
    # The next function is a workaround for the above issue.

    async def aget_embeddings(self, texts: list[str]) -> list[list[float]]:
        if self.backend == "hf":
            # No async support, just call sync
            return self.get_embeddings(texts)
        else:
            # Azure/OpenAI: semaphore logic
            async with self.embedding_semaphore:
                embeddings = await self._async_get_embeddings(texts)

            return embeddings

    async def _async_get_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self.get_embeddings(texts)

    def get_embed_dimension(self) -> int:
        return self.embed_dimension
