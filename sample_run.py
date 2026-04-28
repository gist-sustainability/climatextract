from climatextract import extract_and_evaluate
from climatextract.adapters.azure_ai_foundry import (
    AzureAIFoundryEmbeddingHandler,
    AzureAIFoundryLlmHandler,
)


# Both LLM and embedding live on Azure AI Foundry.
llm = AzureAIFoundryLlmHandler()
embedder = AzureAIFoundryEmbeddingHandler()

result = extract_and_evaluate(
    llm=llm,
    embedder=embedder,
)

print(f"Done — results at {result}")
