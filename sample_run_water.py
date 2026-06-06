"""Sample run for the generic indicator path (water extraction).

Unlike ``sample_run.py`` (CO2 emissions + evaluation), this demonstrates the
additive ``indicator=`` path: the LLM bootstraps an extraction spec for the
named indicator, which drives both the retrieval query and the extraction
prompt. No gold standard is required, so this uses ``extract`` (not
``extract_and_evaluate``).

Swap ``indicator`` for "energy consumption", "waste", etc. to try other
indicators — no code changes needed.
"""

from climatextract import extract
from climatextract.adapters.azure_ai_foundry import (
    AzureAIFoundryEmbeddingHandler,
    AzureAIFoundryLlmHandler,
)


# Both LLM and embedding live on Azure AI Foundry.
llm = AzureAIFoundryLlmHandler()
embedder = AzureAIFoundryEmbeddingHandler()

# Two reports that disclose water volumes in standard GRI 303 tables, so the
# run produces results out of the box. Replace with your own PDFs or omit
# ``pdf_input`` to use ``filename_list`` from climatextract.toml.
result = extract(
    pdf_input=[
        "data/pdfs/hudbay minerals inc_2020_report.pdf",
        "data/pdfs/smith (ds) plc_2021_report.pdf",
    ],
    indicator="water consumption",
    indicator_description=(
        "absolute company-wide water volumes the company reports: water "
        "withdrawal, water consumption and water discharge, in volume units "
        "such as megalitres or cubic metres"
    ),
    llm=llm,
    embedder=embedder,
    verbose=True,
)

print(f"Done — results at {result}")
