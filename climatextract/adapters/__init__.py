"""Reference provider adapters shipped with ClimXtract.

Users on Azure OpenAI / Azure AI Foundry can import handlers from
``climatextract.adapters.azure_openai`` rather than writing their own.
Users on other providers are expected to implement their own
``LlmHandler`` / ``EmbeddingModelHandler`` subclass.
"""
