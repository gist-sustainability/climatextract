"""Unit tests for the additive generic-indicator extraction path.

These are self-contained: no network / LLM / litellm calls. The spec-bootstrap
LLM is replaced with a fake whose ``run_llm`` returns canned JSON.
"""

import asyncio
import json

import pandas as pd
import pytest

from climatextract.indicators.spec import (
    IndicatorEntry,
    IndicatorSpec,
)
from climatextract.indicators.meta_prompt import build_meta_prompt
from climatextract.indicators import spec_bootstrap
from climatextract.indicators.generic_prompt import GenericStructuredJsonPrompt
from climatextract.pipeline import prepare_generic_long_format_from


class _PromptParams:
    """Minimal stand-in for LLMParams (only year_min/year_max are read)."""
    year_min = 2018
    year_max = 2022


# ---------------------------------------------------------------------------
# IndicatorEntry schema — free-string category, value validation
# ---------------------------------------------------------------------------

def test_indicator_entry_accepts_any_category():
    # Unlike the CO2 KpiEntry Literal, any label is accepted (not dropped).
    e = IndicatorEntry(year=2021, category="rainwater harvested", value=12.5, unit="megalitres")
    assert e.category == "rainwater harvested"


def test_indicator_entry_rejects_negative_value():
    with pytest.raises(Exception):
        IndicatorEntry(year=2021, category="withdrawal", value=-5, unit="ML")


def test_indicator_entry_all_nulls_ok():
    e = IndicatorEntry()
    assert e.year is None and e.category is None and e.value is None and e.unit is None


# ---------------------------------------------------------------------------
# Meta-prompt rendering
# ---------------------------------------------------------------------------

def test_meta_prompt_contains_request_and_examples():
    p = build_meta_prompt("water consumption", "net freshwater consumed")
    assert "water consumption" in p
    assert "net freshwater consumed" in p
    # both worked examples are present to anchor the pattern
    assert "carbon emissions" in p and '"Waste"' in p
    assert "search_query" in p
    # no leftover placeholder tokens
    assert "<<INDICATOR_NAME>>" not in p and "<<INDICATOR_DESCRIPTION>>" not in p


def test_meta_prompt_default_description():
    p = build_meta_prompt("energy consumption")
    assert "none provided" in p


# ---------------------------------------------------------------------------
# Spec bootstrap (with a fake LLM)
# ---------------------------------------------------------------------------

_SPEC_JSON = {
    "canonical_name": "Water consumption",
    "definition": "Net freshwater consumed (withdrawal not returned).",
    "subcategories": ["withdrawal", "discharge", "consumption"],
    "synonyms": ["water use"],
    "typical_units": ["megalitres", "m3"],
    "exclusions": ["water intensity"],
    "search_query": "What is the company's total water consumption, withdrawal and discharge per year?",
}


class _FakeLlm:
    def __init__(self, content):
        self._content = content
        self.calls = 0

    def run_llm(self, prompt):
        self.calls += 1
        return {"content": self._content, "logprobs": None}, None


def test_bootstrap_spec_parses_and_persists(tmp_path):
    fake = _FakeLlm(json.dumps(_SPEC_JSON))
    spec = spec_bootstrap.bootstrap_spec(
        "water consumption", "net freshwater", fake, cache_dir=str(tmp_path))

    assert isinstance(spec, IndicatorSpec)
    assert spec.canonical_name == "Water consumption"
    assert spec.subcategories == ["withdrawal", "discharge", "consumption"]
    assert spec.raw_name == "water consumption"
    assert spec.description == "net freshwater"
    assert (tmp_path / spec_bootstrap.SPEC_FILENAME).exists()


def test_bootstrap_spec_reuses_cache(tmp_path):
    spec_bootstrap.bootstrap_spec(
        "water", None, _FakeLlm(json.dumps(_SPEC_JSON)), cache_dir=str(tmp_path))
    # A second call must NOT hit the LLM again.
    fake2 = _FakeLlm("SHOULD NOT BE PARSED")
    spec2 = spec_bootstrap.bootstrap_spec("water", None, fake2, cache_dir=str(tmp_path))
    assert fake2.calls == 0
    assert spec2.canonical_name == "Water consumption"


def test_bootstrap_spec_strips_markdown_fences(tmp_path):
    fenced = "```json\n" + json.dumps(_SPEC_JSON) + "\n```"
    spec = spec_bootstrap.bootstrap_spec("water", None, _FakeLlm(fenced), cache_dir=str(tmp_path))
    assert spec.canonical_name == "Water consumption"


def test_bootstrap_spec_tolerates_surrounding_prose(tmp_path):
    noisy = "Here is the spec you asked for:\n" + json.dumps(_SPEC_JSON) + "\nHope that helps!"
    spec = spec_bootstrap.bootstrap_spec("water", None, _FakeLlm(noisy), cache_dir=str(tmp_path))
    assert spec.search_query.startswith("What is the company's total water")


# ---------------------------------------------------------------------------
# GenericStructuredJsonPrompt
# ---------------------------------------------------------------------------

def _make_spec(subcats=None):
    return IndicatorSpec(
        canonical_name="Water consumption",
        definition="Net freshwater consumed by the company.",
        subcategories=["withdrawal", "discharge"] if subcats is None else subcats,
        synonyms=[],
        typical_units=["megalitres"],
        exclusions=["water intensity"],
        search_query="...",
        raw_name="water",
        description=None,
    )


def test_generic_query_renders_spec_fields():
    gp = GenericStructuredJsonPrompt(_make_spec(), _PromptParams())
    assert "Water consumption" in gp.query
    assert "megalitres" in gp.query
    assert "withdrawal" in gp.query
    assert "water intensity" in gp.query
    assert "{context_str}" in gp.query


def test_generic_prepare_prompt_injects_page_text():
    gp = GenericStructuredJsonPrompt(_make_spec(), _PromptParams())
    out = asyncio.run(gp.prepare_prompt("UNIQUE_PAGE_MARKER_123"))
    assert "UNIQUE_PAGE_MARKER_123" in out


def test_generic_process_output_keeps_unforeseen_category():
    gp = GenericStructuredJsonPrompt(_make_spec(), _PromptParams())
    llm_out = {
        "content": json.dumps({"entries": [
            {"year": 2021, "category": "withdrawal", "value": 12093, "unit": "thousand m3"},
            {"year": 2021, "category": "rainwater harvested", "value": 5.0, "unit": "ML"},
            {"year": 2020, "category": None, "value": 100, "unit": "ML"},
        ]}),
        "logprobs": None,
    }
    df = gp.process_llm_output(llm_out)
    cats = set(df["extracted_scope_from_llm"].dropna().tolist())
    assert "withdrawal" in cats
    assert "rainwater harvested" in cats  # NOT dropped, unlike a closed Literal
    assert 12093 in df["extracted_value_from_llm"].tolist()
    for col in ["scope", "year", "value", "unit", "extracted_scope_from_llm",
                "extracted_year_from_llm", "extracted_value_from_llm",
                "value_probability", "unit_probability", "raw_llm_response"]:
        assert col in df.columns


def test_generic_process_output_falls_back_on_garbage():
    gp = GenericStructuredJsonPrompt(_make_spec(), _PromptParams())
    df = gp.process_llm_output({"content": "this is not json", "logprobs": None})
    # Falls back to the no-extraction grid built from sub-categories.
    assert set(df["scope"].tolist()) >= {"withdrawal", "discharge"}
    assert "extracted_scope_from_llm" in df.columns


def test_fill_no_extractions_atomic_indicator():
    gp = GenericStructuredJsonPrompt(_make_spec(subcats=[]), _PromptParams())
    df = gp._fill_no_extractions_table()
    assert df["scope"].isna().all()
    assert len(df) == (2022 - 2018 + 1)  # one row per year


# ---------------------------------------------------------------------------
# Generic long-format output
# ---------------------------------------------------------------------------

def test_prepare_generic_long_format():
    merged = pd.DataFrame({
        "report_name_short": ["a.pdf", "a.pdf"],
        "extracted_year_from_llm": [2021, 2020],
        "extracted_scope_from_llm": ["withdrawal", None],
        "extracted_value_from_llm": [12093.0, None],  # null-value row must be dropped
        "value_probability": [0.9, None],
        "extracted_unit_from_llm": ["thousand m3", "ML"],
        "unit_probability": [0.8, None],
        "extraction_context": ["ctx1", "ctx2"],
        "page_number_used_by_llm": ["96", "50"],
    })
    out = prepare_generic_long_format_from(merged, "Water consumption")
    assert list(out.columns) == [
        "report_id", "year", "indicator", "category", "value_raw", "value_score",
        "unit_raw", "unit_score", "extraction_context", "page",
    ]
    assert len(out) == 1
    row = out.iloc[0]
    assert row["indicator"] == "Water consumption"
    assert row["category"] == "withdrawal"
    assert row["value_raw"] == 12093.0


# ---------------------------------------------------------------------------
# Guard: the CO2 path is unchanged
# ---------------------------------------------------------------------------

def test_co2_kpi_entry_literal_still_strict():
    """The CO2 schema must remain a closed Literal — proving we did not loosen it."""
    from climatextract.prompts_with_prompt_parsers import KpiEntry

    # A non-scope label is still rejected on the CO2 path.
    with pytest.raises(Exception):
        KpiEntry(kpi_name="withdrawal", year=2021, value=1, unit="ML")

    # Valid GHG scopes still accepted.
    assert KpiEntry(kpi_name="1", year=2021, value=1, unit="tCO2e").kpi_name == "1"
