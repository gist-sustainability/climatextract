"""Data models for generic indicator extraction.

``IndicatorSpec`` is the rendered source of truth for one quantitative
sustainability indicator. It is produced by the LLM spec-bootstrap step
(see ``spec_bootstrap.py``) and rendered into both the semantic-search query
and the extraction prompt (see ``generic_prompt.py``).

``IndicatorEntry`` is the per-value schema the extraction LLM fills. Unlike
the CO2 ``KpiEntry`` (whose ``kpi_name`` is a closed ``Literal`` of GHG
scopes), ``category`` here is a free string: corporate reports use open,
non-standardized sub-category labels across indicators, so anything the model
reads is captured verbatim rather than rejected at validation time.
"""

from dataclasses import asdict, dataclass, field
from typing import List, Optional

from pydantic import BaseModel, Field, confloat


@dataclass
class IndicatorSpec:
    """Rendered specification for one indicator. Source of truth for a run."""

    canonical_name: str
    definition: str
    subcategories: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)
    typical_units: List[str] = field(default_factory=list)
    exclusions: List[str] = field(default_factory=list)
    search_query: str = ""
    # Provenance: what the user actually asked for.
    raw_name: str = ""
    description: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_model(
        cls,
        model: "IndicatorSpecModel",
        raw_name: str,
        description: Optional[str] = None,
    ) -> "IndicatorSpec":
        """Build a spec from a parsed meta-prompt response plus provenance."""
        return cls(
            canonical_name=model.canonical_name,
            definition=model.definition,
            subcategories=list(model.subcategories or []),
            synonyms=list(model.synonyms or []),
            typical_units=list(model.typical_units or []),
            exclusions=list(model.exclusions or []),
            search_query=model.search_query,
            raw_name=raw_name,
            description=description,
        )


class IndicatorSpecModel(BaseModel):
    """Pydantic schema the meta-prompt output is parsed into."""

    canonical_name: str = Field(description="The standard name for this indicator.")
    definition: str = Field(
        description="Precise statement of exactly what is and is not counted."
    )
    subcategories: List[str] = Field(
        default_factory=list,
        description="Breakdowns reports typically disclose, using report-language labels.",
    )
    synonyms: List[str] = Field(
        default_factory=list,
        description="Alternate phrasings and older standard aliases.",
    )
    typical_units: List[str] = Field(
        default_factory=list,
        description="Units the indicator is reported in, as written in reports.",
    )
    exclusions: List[str] = Field(
        default_factory=list,
        description="Value types to ignore (intensity ratios, percentages, etc.).",
    )
    search_query: str = Field(
        description="A natural-language question for semantic retrieval over report pages."
    )


class IndicatorEntry(BaseModel):
    """One extracted absolute value. ``category`` is free-form by design."""

    year: Optional[int] = Field(
        None, description="Reporting year of the value, e.g. 2021.", example=2021
    )
    category: Optional[str] = Field(
        None,
        description=(
            "Sub-category label exactly as written in the report "
            "(e.g. 'withdrawal', 'Scope 1', 'renewable electricity', "
            "'waste to landfill'), or null if the value is not broken down."
        ),
        example="withdrawal",
    )
    value: Optional[confloat(ge=0)] = Field(
        None,
        description="The absolute numeric value, or null if not available.",
        example=12093.0,
    )
    unit: Optional[str] = Field(
        None,
        description="Unit as written in the report (e.g. 'megalitres', 'GJ', 't'), or null.",
        example="megalitres",
    )


class IndicatorEntries(BaseModel):
    """A list of extracted entries (the extraction LLM's JSON output)."""

    entries: List[IndicatorEntry] = Field(
        default_factory=list, description="List of extracted indicator entries."
    )
