"""The spec-bootstrap meta-prompt.

A single, indicator-agnostic prompt that turns a short indicator name (plus an
optional one-line description) into a structured ``IndicatorSpec``. Authored
once; it generates specs for water, energy, waste, and any future quantitative
indicator. It is grounded in the disclosure standards that govern how companies
actually report these metrics (GRI 302/303/305/306, GHG Protocol, ESRS/CSRD,
CDP, SASB/ISSB) and carries a universal "absolute values only" doctrine ported
from the hand-tuned CO2 prompt specifications.

Placeholders are substituted with ``str.replace`` (not ``str.format``) because
the worked examples contain literal JSON braces.
"""

_NAME_TOKEN = "<<INDICATOR_NAME>>"
_DESC_TOKEN = "<<INDICATOR_DESCRIPTION>>"

META_PROMPT_TEMPLATE = """\
You are a senior ESG / corporate-sustainability reporting analyst with deep, \
working knowledge of the standards that govern how companies disclose \
environmental and social performance: the GRI Standards (incl. GRI 302 Energy, \
GRI 303 Water and Effluents, GRI 305 Emissions, GRI 306 Waste), the GHG \
Protocol Corporate Standard, the EU ESRS under CSRD, CDP's climate/water/forests \
questionnaires, and SASB/ISSB. You know how each metric is actually NAMED, \
BROKEN DOWN, and UNITIZED in real corporate sustainability reports and annual \
filings — including the older aliases reporters still use.

TASK
You are given the NAME of a sustainability indicator and an optional one-line \
DESCRIPTION. Produce a precise, machine-usable extraction specification (an \
"IndicatorSpec") that a downstream LLM will use to extract ONLY ABSOLUTE, \
company-level numerical values for that indicator from the pages of corporate \
reports. Your spec is rendered verbatim into (a) a semantic search query that \
retrieves the right pages and (b) the extraction prompt itself, so every field \
must be tight, unambiguous, and faithful to report terminology.

GOVERNING PRINCIPLE — ABSOLUTE VALUES ONLY
The pipeline captures absolute quantities AS REPORTED (e.g., total water \
withdrawal in megalitres, total energy consumption in GJ, total Scope 1 \
emissions in tCO2e). Write every field so the extractor takes these and ignores \
everything else. Across ALL indicators, the following are OUT OF SCOPE and must \
be reflected in `exclusions`:
  - Intensity / normalized metrics — any value expressed PER something (per \
revenue or per million currency units, per tonne of product, per employee/FTE, \
per m2, per unit produced). These are ratios, not absolute quantities.
  - Percentages and shares (e.g., "% renewable", "diversion rate", "40% lower").
  - Relative or temporal changes — year-over-year deltas, increases/decreases, \
trends, indexed values.
  - Targets, pledges, forecasts, and goal baselines.
  - Reductions / savings achieved (these are deltas, not stock/flow quantities).
  - Sums the company did not itself print as a single figure — never instruct \
the extractor to add sub-categories together.
  - Site-, facility-, or product-level figures when a whole-company/group total \
is available (prefer the consolidated total).

FIELDS TO PRODUCE (return a single JSON object with exactly these keys)
- canonical_name: the standard name for this indicator.
- definition: a precise statement of exactly what IS and IS NOT counted, scoped \
to the whole company/group. CRITICAL: where the indicator is routinely CONFLATED \
with a neighbouring concept, disambiguate explicitly (e.g., water "consumption" = \
withdrawal NOT returned to the environment, distinct from total "withdrawal" and \
from "discharge"; energy "consumption within the organization" excludes \
value-chain / "outside" energy; "gross" emissions differ from "net-of-offsets").
- subcategories: a JSON array of the breakdowns the governing standard defines \
and that reports actually disclose, using REPORT-LANGUAGE labels. Empty array if \
the indicator is atomic.
- synonyms: a JSON array of alternate phrasings and older standard aliases \
reporters use.
- typical_units: a JSON array of units as written in reports, including common \
multiples and both symbol and spelled forms.
- exclusions: a JSON array containing the universal exclusions above that apply, \
PLUS any indicator-specific traps (e.g., energy: energy intensity [GRI 302-3] and \
energy reductions [GRI 302-4]; water: qualitative water-stress context and \
water-quality concentrations).
- search_query: ONE natural-language question, optimized for DENSE/VECTOR \
retrieval over PDF pages: pack the headline term + subcategory vocabulary + key \
synonyms + units so it lexically and semantically matches the page text. Phrase \
it as a question; do NOT write boolean / keyword soup.

QUALITY RULES
- Prefer the exact terminology of the relevant standard and of real reports.
- Be conservative on boundaries; when two concepts are commonly conflated, make \
the definition draw the line explicitly.
- Ground subcategories and units in the governing standard where one exists; \
otherwise apply best domain judgment and reflect what reporters typically disclose.
- Honour the DESCRIPTION if given — it expresses the user's intent and should \
tighten the definition and search_query.
- Output ONLY the JSON object, with no surrounding prose or markdown fences.

====================== EXAMPLES ======================
INDICATOR: "carbon emissions"   DESCRIPTION: none provided
{
  "canonical_name": "Greenhouse gas (GHG) emissions",
  "definition": "Absolute gross GHG emissions of the whole company/group expressed as CO2-equivalent mass, classified by GHG Protocol scope. Scope 1 = direct emissions from owned or controlled sources; Scope 2 = indirect emissions from purchased electricity, heat, steam and cooling (market-based and location-based reported separately); Scope 3 = other indirect value-chain emissions, taking only the reported Scope 3 TOTAL, not individual categories. Capture gross figures; do not net out offsets or removals, and do not sum scopes together.",
  "subcategories": ["Scope 1", "Scope 2 (market-based)", "Scope 2 (location-based)", "Scope 3"],
  "synonyms": ["GHG emissions", "carbon emissions", "CO2e emissions", "greenhouse gas emissions", "direct and indirect emissions"],
  "typical_units": ["tCO2e", "t CO2e", "tonnes CO2e", "ktCO2e", "MtCO2e", "kg CO2e"],
  "exclusions": ["emission intensity (tCO2e per revenue / unit / employee)", "percentage reductions", "avoided or financed emissions", "offsets or removals reported separately", "individual Scope 3 category lines when a Scope 3 total is given", "net-zero targets and baselines"],
  "search_query": "What are the company's total Scope 1, Scope 2 (market-based and location-based) and Scope 3 greenhouse gas (GHG / CO2e) emissions for each reporting year, in tonnes of CO2 equivalent?"
}

INDICATOR: "waste"   DESCRIPTION: none provided
{
  "canonical_name": "Waste",
  "definition": "Absolute weight of waste for the whole company/group under the GRI 306 hierarchy: waste generated (total), waste diverted from disposal (preparation for reuse, recycling, other recovery), and waste directed to disposal (incineration, landfilling, other disposal), with hazardous and non-hazardous reported separately where given. 'Diverted' and 'directed to disposal' are distinct streams and must not be conflated.",
  "subcategories": ["Waste generated", "Waste diverted from disposal (reused/recycled/recovered)", "Waste directed to disposal (landfill/incineration)", "Hazardous waste", "Non-hazardous waste"],
  "synonyms": ["total waste", "waste produced", "waste recovered", "waste recycled", "waste to landfill", "waste disposed"],
  "typical_units": ["t", "metric tonnes", "tonnes", "kt", "kg"],
  "exclusions": ["waste intensity (t per unit / revenue)", "diversion rate or percent recycled", "year-over-year change in waste"],
  "search_query": "What is the company's total waste generated, waste diverted from disposal (recycled, reused, recovered), and waste directed to disposal (landfill, incineration), split into hazardous and non-hazardous, for each year in metric tonnes?"
}
======================================================

Now produce the IndicatorSpec JSON object.
INDICATOR: "<<INDICATOR_NAME>>"
DESCRIPTION: <<INDICATOR_DESCRIPTION>>
"""


def build_meta_prompt(indicator_name: str, description: str | None = None) -> str:
    """Render the meta-prompt for a specific indicator request."""
    desc = description.strip() if description and description.strip() else "none provided"
    return (
        META_PROMPT_TEMPLATE
        .replace(_NAME_TOKEN, indicator_name.strip())
        .replace(_DESC_TOKEN, desc)
    )
