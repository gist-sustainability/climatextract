"""Bootstrap an ``IndicatorSpec`` from a short indicator name via one LLM call.

The user types an indicator name (and optional one-line description); this
module renders the meta-prompt, asks the LLM, parses the JSON into an
``IndicatorSpec``, and persists it to the run directory (per-run, git-ignored)
so every run is reproducible and the analyst can eyeball/override the spec.
"""

import json
import logging
import os
from typing import Optional

from climatextract.indicators._json import extract_json_object
from climatextract.indicators.meta_prompt import build_meta_prompt
from climatextract.indicators.spec import IndicatorSpec, IndicatorSpecModel

logger = logging.getLogger(__name__)

SPEC_FILENAME = "_indicator_spec.json"


def bootstrap_spec(
    indicator_name: str,
    indicator_description: Optional[str],
    llm,
    cache_dir: Optional[str] = None,
) -> IndicatorSpec:
    """Generate (or reload) the ``IndicatorSpec`` for an indicator request.

    Args:
        indicator_name: Short indicator name, e.g. ``"water consumption"``.
        indicator_description: Optional one-line clarification of intent.
        llm: A ``climatextract.llm_embedding_api_bridge.Llm`` wrapper. The same
            handler used for extraction generates the spec.
        cache_dir: Run directory. If it already contains a spec JSON, that is
            reused (idempotent within a run); otherwise the new spec is written
            there.

    Returns:
        The resolved ``IndicatorSpec``.
    """
    if cache_dir:
        cached = _load_cached_spec(cache_dir)
        if cached is not None:
            logger.info("Reusing cached indicator spec from %s", cache_dir)
            return cached

    prompt = build_meta_prompt(indicator_name, indicator_description)
    response_dict, error = llm.run_llm(prompt)
    if error is not None:
        raise RuntimeError(f"Spec bootstrap LLM call failed: {error}") from error

    content = (response_dict or {}).get("content", "")
    if not content.strip():
        raise RuntimeError("Spec bootstrap returned empty content from the LLM.")

    data = extract_json_object(content)
    model = IndicatorSpecModel(**data)
    spec = IndicatorSpec.from_model(
        model, raw_name=indicator_name, description=indicator_description
    )

    if cache_dir:
        _persist_spec(spec, cache_dir)

    return spec


def _load_cached_spec(cache_dir: str) -> Optional[IndicatorSpec]:
    path = os.path.join(cache_dir, SPEC_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return IndicatorSpec(**data)
    except (OSError, TypeError, ValueError) as e:
        logger.warning("Could not load cached indicator spec (%s); regenerating.", e)
        return None


def _persist_spec(spec: IndicatorSpec, cache_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, SPEC_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec.to_dict(), f, indent=2, ensure_ascii=False)


def format_spec_for_console(spec: IndicatorSpec) -> str:
    """A compact, human-readable rendering of the spec for run output."""
    lines = [
        f"Indicator spec (from \"{spec.raw_name}\""
        + (f" / \"{spec.description}\"" if spec.description else "")
        + "):",
        f"  canonical_name: {spec.canonical_name}",
        f"  definition:     {spec.definition}",
        f"  subcategories:  {', '.join(spec.subcategories) or '(none)'}",
        f"  typical_units:  {', '.join(spec.typical_units) or '(none)'}",
        f"  exclusions:     {', '.join(spec.exclusions) or '(none)'}",
        f"  search_query:   {spec.search_query}",
    ]
    return "\n".join(lines)
