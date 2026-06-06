"""Generic, additive support for extracting non-CO2 sustainability indicators.

Only exercised when ``extract()`` is called with an ``indicator=`` argument; the
CO2 path is untouched. An ``IndicatorSpec`` (spec.py), bootstrapped by the LLM
from a short name (meta_prompt.py / spec_bootstrap.py), drives both the retrieval
query and ``GenericStructuredJsonPrompt`` (generic_prompt.py) — no per-indicator
code needed.
"""

from climatextract.indicators.spec import (
    IndicatorEntries,
    IndicatorEntry,
    IndicatorSpec,
    IndicatorSpecModel,
)

__all__ = [
    "IndicatorSpec",
    "IndicatorSpecModel",
    "IndicatorEntry",
    "IndicatorEntries",
]
