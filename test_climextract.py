
#########################################################################
# Proposal for an improved interface
#########################################################################
from climatextract import extract

# Current interface
result_path = extract("data/pdfs/inchcape plc_2022_report.pdf")
print(f"Results saved to: {result_path}")

# Alternative suggestion:  Possible future interface. The user will need to create an embedder object and an LLM object 

# See below for how a user would need to specify EmbeddingModel, using litellm.
# embedder = EmbeddingModel()
# llm = NotImplemented

# not (fully) implemented. We would replace config.py with llm_embedding_api_bridge.py
# result_path = extract("data/pdfs/inchcape plc_2022_report.pdf", embedder, llm)


#########################################################################
# One possible implementation of the EmbeddingModel class
# We would want the user to specify this class, implementing methods from llm_embedding_api_bridge.EmbeddingModelHandler
#########################################################################

import asyncio
import os
from typing import Tuple

import litellm
from climatextract import llm_embedding_api_bridge


class EmbeddingModel(llm_embedding_api_bridge.EmbeddingModelHandler):

    def __init__(self, model="text-embedding-3-large", model_dict=None):
        """You should change the __init()__ function, especially the model dict,
        in a way that this class runs on your machine.

        See LiteLLm documentation at https://docs.litellm.ai/docs/embedding/supported_embedding
        Coupling between Climatextract and LiteLLM is purposefully loose. Climateextract just expects 
        
        Args:
            model: A string specifying the model to use. Ignored if model_dict is provided.
            model_dict: A dictionary containing the model configuration.
        """
        
        if model_dict:
            self.model_dict = model_dict

        else:

            try:
                from azure_authentication import customized_azure_login
                credential = customized_azure_login.CredentialFactory().select_credential()
                login_token_provider = llm_embedding_api_bridge.ThreadSafeTokenProvider(credential)
            except Exception as e:
                credential = None

            match model:
                case "text-embedding-ada-002":
                    self.model_dict ={"model": "azure/text-embedding-ada-002", # or azure/text-embedding-3-large
                                    "api_base": os.environ["AZURE_ENDPOINT"], 
                                    "api_version": os.environ["API_VERSION"],
                                    "azure_ad_token_provider": login_token_provider,
                                    "max_concurrent_calls": 2 # requires testing: values other than 1 did not work properly in llama index (no support for async embedding calls there)
                                    }
                case "text-embedding-3-large":
                    self.model_dict ={"model": "azure/text-embedding-3-large",
                                    "api_base": os.environ["AZURE_ENDPOINT"], 
                                    "api_version": os.environ["API_VERSION"],
                                    "azure_ad_token_provider": login_token_provider,
                                    "max_concurrent_calls": 2 # requires testing: values other than 1 did not work properly in llama index (no support for async embedding calls there)
                                    }
                    
                case _:
                    raise ValueError(f"Model {model} not supported. Please provide a model_dict with the necessary configuration.")
    
    def get_model_dict(self):
        return self.model_dict
    
    def get_max_concurrent_calls(self):
        if self.model_dict and "max_concurrent_calls" in self.model_dict:
            return self.model_dict["max_concurrent_calls"]
        else:
            return 1

    def get_embedding(self, texts: list[str]) -> Tuple[litellm.EmbeddingResponse, float]:
        response = litellm.embedding(input=texts, **self.model_dict)
        costs = litellm.completion_cost(response)
        return response, costs

    async def aget_embedding(self, texts: list[str]) -> Tuple[litellm.EmbeddingResponse, float]:
        response = await litellm.aembedding(input=texts, **self.model_dict)
        costs = litellm.completion_cost(response)
        return response, costs
    
    def test_embedding_setup(self):
        """"Check that your setup and connection works by running this function."""

        response, costs = self.get_embedding(["test string", "another test string"])
        print(f"Test successful!! Your embedding model is working.")
        print(f"It is configured to run at most {self.get_max_concurrent_calls()} concurrent calls in parallel to avoid RateLimitErrors.")
        print(f"Your embedding model returns a vector of length {len(response.data[0]['embedding'])} for each input text.")
        print(f"LiteLLM fetches the community-maintained model cost map at import time from https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json ")
        print(f"Incured costs from this call: ${float(costs):.10f} (according to LiteLLM's cost map.)")
        


# Test embedding model
EmbeddingModel().test_embedding_setup()
embedder = EmbeddingModel()

usagecounter = llm_embedding_api_bridge.UsageCounter()
embed_model = llm_embedding_api_bridge.EmbeddingModel(embedder, usage_counter=usagecounter)
print(embed_model.get_model_name())
print(embed_model.get_embed_dimension())
print(usagecounter.get_usage_dict())


##############################################################
# Some additional preliminary code.
# 1. LiteLLM provides callback functions -> not needed?! We have our own solution with the UsageCounter.
# 2. Some minimal code to run our LLMs with litellm. Looks very promising, and one would need to put this into an LlmModel class, similar to EmbeddingModel.
##############################################################

# def custom_callback(kwargs, completion_response, start_time, end_time):
#     try:
#         model = kwargs.get("model")
#         prompt_tokens = kwargs.get("prompt_tokens")
#         completion_tokens = kwargs.get("completion_tokens")
#         total_tokens = kwargs.get("total_tokens")
#         cost = kwargs.get("response_cost", 0)
#         print(f"Model: {model}")
#         print(f"Prompt Tokens: {prompt_tokens}")
#         print(f"Completion Tokens: {completion_tokens}")
#         print(f"Total Tokens: {total_tokens}")
#         print(f"Cost: ${cost:.6f}")
#     except:
#         pass

# litellm.success_callback = [custom_callback]

# response = litellm.completion(
#   "azure/gpt-4o-mini-2024-07-18",
#   messages = [{ "content": "Hello, how are you?","role": "user"}],
#   api_base=os.environ["AZURE_ENDPOINT"],
#   api_version=os.environ["API_VERSION"],
#   azure_ad_token_provider=login_token_provider
# )

# print(response)
# print(response.choices[0].message.content)


# Some experiments with calling the LLMs via the litellm package directly. Looks very promising. 

try:
    from azure_authentication import customized_azure_login
    credential = customized_azure_login.CredentialFactory().select_credential()
    login_token_provider = llm_embedding_api_bridge.ThreadSafeTokenProvider(credential)
except Exception as e:
    credential = None

async def test_get_response(user_message = "Hello, how are you?"):
    messages = [{"content": user_message, "role": "user"}]
    response = await litellm.acompletion(model="azure/gpt-4o-mini-2024-07-18", 
                                 messages=messages,
                                 api_base=os.environ["AZURE_ENDPOINT"],
                                 api_version=os.environ["API_VERSION"],
                                 azure_ad_token_provider=login_token_provider)
    print(f"Cost: ${float(litellm.completion_cost(response)):.10f}")
    print(f"Completion tokens: {response.usage.completion_tokens}")
    print(f"Prompt tokens: {response.usage.prompt_tokens}")
    print(f"Total tokens: {response.usage.total_tokens}")
    print(response.model)
    # print(litellm.token_counter(response))
    return response

print("Running azure/gpt-4o-mini-2024-07-18 ...")
response = asyncio.run(test_get_response())
print(response)

async def test_get_response(user_message = "Hello, how are you?"):
    messages = [{"content": user_message, "role": "user"}]
    response = await litellm.acompletion(model="azure_ai/gpt-oss-120b", 
                                 messages=messages,
                                 api_base=os.environ["AZURE_AI_FOUNDRY_ENDPOINT"] + "openai/v1",
                                 # api_version="2025-01-01-preview", not needed!!!
                                 azure_ad_token_provider=login_token_provider)
    print(f"Cost: ${float(litellm.completion_cost(response)):.10f}")
    print(f"Completion tokens: {response.usage.completion_tokens}")
    print(f"Prompt tokens: {response.usage.prompt_tokens}")
    print(f"Total tokens: {response.usage.total_tokens}")
    print(response.model)
    # print(litellm.token_counter(response))
    return response

print("Running azure_ai/gpt-oss-120b ...")
response = asyncio.run(test_get_response())
print(response)