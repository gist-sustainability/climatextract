"""Tolerant JSON-object extraction from LLM responses.

Shared by the spec bootstrap and the generic extraction parser so both handle
markdown code fences and surrounding prose the same way.
"""

import json
import re

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def extract_json_object(text: str) -> dict:
    """Parse the first balanced JSON object out of an LLM response.

    Tolerates markdown code fences, leading/trailing prose, and ``{{ }}``
    brace-doubling (models occasionally echo the escaped braces from the
    prompt's format instructions). Raises ``ValueError`` if no parseable
    object is present.
    """
    cleaned = _FENCE_RE.sub("", text).strip()
    try:
        return _parse(cleaned)
    except ValueError:
        collapsed = cleaned.replace("{{", "{").replace("}}", "}")
        if collapsed == cleaned:
            raise
        return _parse(collapsed)


def _parse(cleaned: str) -> dict:
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fall back to the first balanced {...} span.
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM response.")
    depth = 0
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(cleaned[start : i + 1])
    raise ValueError("Unbalanced JSON object in LLM response.")
