"""Generic structured-JSON prompt processor for arbitrary indicators.

Mirrors ``StructuredJsonPrompt`` (the CO2 structured-JSON path) but renders its
prompt from an ``IndicatorSpec`` and parses output into the free-string-category
``IndicatorEntries`` schema: the category label is carried through verbatim, so a
never-before-seen sub-category is captured rather than dropped. Output columns
match ``StructuredJsonPrompt`` so the rest of the pipeline is unchanged.
"""

import logging
from typing import List

import pandas as pd
from llama_index.core import PromptTemplate
from llama_index.core.output_parsers import PydanticOutputParser
from pydantic import ValidationError

import climatextract.helpers as helpers
from climatextract.indicators._json import extract_json_object
from climatextract.indicators.spec import IndicatorEntry, IndicatorEntries, IndicatorSpec
from climatextract.prompts_with_prompt_parsers import PromptProcessorInterface

logger = logging.getLogger(__name__)

# Universal "absolute values only" doctrine, indicator-agnostic. Mirrors the
# spirit of the CO2 PromptSpecifications but parameterized by the spec.
_UNIVERSAL_SPECIFICATIONS = (
    "Only extract values which refer to the whole company or group.\n"
    "Only extract absolute values reported for the indicator (a quantity with a unit).\n"
    "Do not extract intensity or normalized values (anything expressed 'per' something, "
    "e.g. per unit of revenue, per tonne of product, per employee, per square metre).\n"
    "Do not extract percentages, shares, ratios, or relative year-over-year changes.\n"
    "Do not extract targets, forecasts, pledges, or reductions/savings achieved.\n"
    "Footnotes or annotations in metric names should be treated as references and ignored.\n"
    "Do not perform any calculations or transformations on the values. Extract and report "
    "the data exactly as presented. Do not invent values and do not sum sub-categories.\n"
)


class GenericStructuredJsonPrompt(PromptProcessorInterface):
    """Spec-driven structured-JSON extraction for any quantitative indicator."""

    def __init__(self, spec: IndicatorSpec, prompt_params):
        self.spec = spec
        self.min_year = prompt_params.year_min if prompt_params.year_min is not None else 2010
        self.max_year = prompt_params.year_max if prompt_params.year_max is not None else 2024

        self.query = self._build_query(spec)
        self.parser = PydanticOutputParser(output_cls=IndicatorEntries)

    def _build_query(self, spec: IndicatorSpec) -> str:
        role = (
            "You are a sustainability-reporting analyst tasked with extracting specific "
            "absolute numerical data from corporate reports.\n"
            f"Your objective is to extract only the absolute values for the indicator "
            f"\"{spec.canonical_name}\" across the entire company.\n\n"
        )

        definition = f"INDICATOR DEFINITION:\n{spec.definition}\n\n"

        if spec.subcategories:
            subcats = (
                "SUB-CATEGORIES:\n"
                "Capture each reported value and label it with the sub-category as written in "
                "the report. Common sub-categories for this indicator include: "
                + "; ".join(spec.subcategories)
                + ".\nIf a value does not match any of these, record the label exactly as it "
                "appears in the report rather than discarding it. If a value is a single overall "
                "total with no breakdown, leave the category null.\n\n"
            )
        else:
            subcats = (
                "SUB-CATEGORIES:\n"
                "Label each value with the sub-category as written in the report, or leave the "
                "category null if it is a single overall total.\n\n"
            )

        units = ""
        if spec.typical_units:
            units = (
                "EXPECTED UNITS (report the unit exactly as written; these are typical): "
                + ", ".join(spec.typical_units)
                + ".\n"
            )

        specifications = "SPECIFICATIONS:\n" + _UNIVERSAL_SPECIFICATIONS + units
        if spec.exclusions:
            specifications += "Additionally, do NOT extract: " + "; ".join(spec.exclusions) + ".\n"
        specifications += "\n"

        year_range = (
            f"Year range for the search: only extract values from {self.min_year} to "
            f"{self.max_year}.\n\n"
        )

        return (
            f"{role}{definition}{subcats}{specifications}{year_range}"
            "Here is the excerpt: \n {context_str}"
        )

    async def prepare_prompt(self, doc_text: str) -> str:
        prompt_tmpl = PromptTemplate(template=self.query, output_parser=self.parser)
        return prompt_tmpl.format(context_str=doc_text)

    def process_llm_output(self, llm_output) -> pd.DataFrame:
        """Parse JSON output and compute confidence if log-probs are available."""
        if isinstance(llm_output, dict):
            raw_output_str = str(llm_output.get("content", ""))
            logprobs_object = llm_output.get("logprobs")
            log_blocks = logprobs_object.content if logprobs_object else None
        else:
            raw_output_str, log_blocks = str(llm_output), None

        try:
            content = extract_json_object(raw_output_str)
            valid_entries = self._validate_llm_output_content(content)
            output_table = self._reformat_output_table(valid_entries, raw_output_str, log_blocks)
        except Exception as e:  # noqa: BLE001 — match CO2 path's defensive parsing
            logger.warning("Error while processing LLM output: %s", e)
            empty_table = self._fill_no_extractions_table()
            output_table = self._reformat_output_table(empty_table, raw_output_str, log_blocks)

        return output_table

    def _validate_llm_output_content(self, llm_output) -> List[dict]:
        """Validate each entry; keep originals for confidence calculation."""
        valid_entries = []
        for entry in llm_output.get("entries", []):
            try:
                validated_entry = IndicatorEntry(**entry)
                entry_dict = vars(validated_entry)
                entry_dict["value_original"] = entry.get("value")
                entry_dict["unit_original"] = entry.get("unit")
                valid_entries.append(entry_dict)
            except ValidationError as e:
                logger.warning("Validation error for entry %s: %s", entry, e)
        return valid_entries

    def _reformat_output_table(self, output_list: List, llm_output: str, log_blocks=None) -> pd.DataFrame:
        """Change output table to the column format the pipeline expects."""
        if len(output_list) > 0:
            output_table = pd.DataFrame(output_list)
            # Carry the free-string category through as 'scope' (identity map).
            output_table = output_table.rename(columns={"category": "scope"})
        else:
            output_table = self._fill_no_extractions_table()

        # Identity mapping: the category label is the extracted dimension.
        output_table["extracted_scope_from_llm"] = output_table["scope"]
        output_table["extracted_year_from_llm"] = pd.to_numeric(
            output_table["year"], errors="coerce"
        )
        output_table["extracted_value_from_llm"] = pd.to_numeric(
            output_table["value"], errors="coerce"
        )

        if log_blocks is not None and len(output_list) > 0:
            output_table["value_probability"] = output_table.apply(
                lambda r: helpers.compute_value_confidence(
                    r.get("value_original", r["value"]), log_blocks
                ),
                axis=1,
            )
            output_table["unit_probability"] = output_table.apply(
                lambda r: helpers.compute_value_confidence(
                    r.get("unit_original", r["unit"]), log_blocks
                ),
                axis=1,
            )
            for col in ("value_original", "unit_original"):
                if col in output_table.columns:
                    output_table = output_table.drop(columns=[col])
        else:
            output_table["value_probability"] = pd.NA
            output_table["unit_probability"] = pd.NA

        output_table["raw_llm_response"] = llm_output
        return output_table

    def _fill_no_extractions_table(self) -> pd.DataFrame:
        """Create a table when nothing was extracted.

        Uses the spec's sub-categories (the soft hints) as the grid dimension,
        falling back to a single unspecified row per year for atomic indicators.
        """
        categories = self.spec.subcategories if self.spec.subcategories else [pd.NA]
        output_table_dict = {
            "year": [str(num) for num in range(self.min_year, self.max_year + 1)],
            "scope": categories,
        }
        output_table = helpers.expand_grid(output_table_dict)
        output_table["value"] = "Nothing extracted. No Regex match"
        output_table["unit"] = "Nothing extracted. No Regex match"
        return output_table
