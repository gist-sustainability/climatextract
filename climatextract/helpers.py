"""
This file provides a mix of helper functions that don't fit well elsewhere.
"""

import asyncio
import itertools
import os
import os.path
import re
import json
import math
import bisect
from functools import lru_cache
from itertools import accumulate

import numpy as np
import pandas as pd

# Default path to the bundled unit normalization CSV file.
# Uses __file__ to locate it relative to this module, works both when running
# from source and when installed as a package
_DEFAULT_UNIT_NORMALIZATION_PATH = os.path.join(
    os.path.dirname(__file__), "data", "unit_normalization_dict.csv"
)
_DEFAULT_UNIT_CLASSIFIER_PATH = os.path.join(
    os.path.dirname(__file__), "data", "unit_harmonization_voting_classifier.joblib"
)


def _extract_quantity_keyword(text: str) -> str:
    """Extract the quantity marker used by the unit normalization model."""
    text = text.lower()
    if re.search(
        r"\b(tsd\.?|thousand|1,?000|10.?3t?|kilo|kt|tt|million kg|tausend|ktons|kiloton(ne)?s?)\b|\bkt(?=[A-Za-z])",
        text,
    ):
        return "kt"
    if re.search(
        r"\b(million(en)?|mio\.?|1,?000,?000|mega|megaton(ne)?s?n?|mm\s*mt|mm|mn|mmt(on)?(ne)?s?)\b",
        text,
    ):
        return "Mt"
    if re.search(r"\b(kg|kilograms?)\b|\bkg(?=[A-Za-z])", text):
        return "kg"
    if re.search(r"\b(us[\s-]?ton(ne)?s?|ust|short tons?)\b", text):
        return "USt"
    if re.search(
        r"\b(billion(s)?|milliarde(n)?|bn|1,?000,?000,?000|giga|gigaton(ne)?s?|gt)\b",
        text,
    ):
        return "Gt"
    return ""


def _clean_unit(text: str) -> str:
    """Remove numeric and connector tokens from a raw unit string."""
    text = re.sub(r"[0-9\.,]+", "", text)
    text = re.sub(r"[^\w]+", " ", text)
    text = re.sub(r"\bof\b", "", text)
    return text.strip().lower()


def _create_unit_classifier_features(units: list[str]) -> pd.DataFrame:
    raw_units = pd.Series(units, dtype="object")
    cleaned_text = raw_units.str.lower().str.replace(
        r"\d+[,\.]?\d*", "NUMBER", regex=True
    )
    return pd.DataFrame(
        {
            "text": (
                cleaned_text
                + " "
                + raw_units.apply(_extract_quantity_keyword)
                + " "
                + raw_units.apply(_clean_unit)
            ),
            "length": raw_units.str.len(),
        }
    )


@lru_cache(maxsize=1)
def _load_unit_classifier(model_path: str):
    """Load and cache the bundled unit classifier."""
    import joblib

    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise ValueError("The unit classifier file must contain a classifier bundle.")
    if "pipeline" not in bundle or "label_encoder" not in bundle:
        raise ValueError(
            "The unit classifier bundle must contain 'pipeline' and 'label_encoder'."
        )
    pipeline = bundle["pipeline"]
    label_encoder = bundle["label_encoder"]
    if not callable(getattr(pipeline, "predict", None)) or not callable(
        getattr(label_encoder, "inverse_transform", None)
    ):
        raise ValueError("The unit classifier bundle contains invalid components.")
    return pipeline, label_encoder


def _predict_unit_normalizations(units: list[str]) -> list[str]:
    pipeline, label_encoder = _load_unit_classifier(_DEFAULT_UNIT_CLASSIFIER_PATH)
    predictions = pipeline.predict(_create_unit_classifier_features(units))
    normalized_units = label_encoder.inverse_transform(predictions).tolist()
    if len(normalized_units) != len(units):
        raise ValueError("The unit classifier returned an unexpected number of labels.")
    if not all(isinstance(unit, str) for unit in normalized_units):
        raise ValueError("The unit classifier returned a non-string unit label.")
    return normalized_units


def _correct_ambiguous_mt_unit(unit: str, value, scope) -> str | None:
    """Return a corrected category for ambiguous ``m t`` units, if needed."""
    if not isinstance(unit, str):
        return None
    if re.search(r"\b(?:millions?|millionen?|mio\.?|mega)", unit, re.IGNORECASE):
        return None
    if not re.search(
        r"\bm\s*\.?\s*(?:t(?=\s|$|[-/().,:;]|co2|co₂)|tons?|tonnes?)",
        unit,
        re.IGNORECASE,
    ):
        return None

    scope_value = str(scope).strip().lower() if pd.notna(scope) else ""
    if scope_value == "3":
        threshold = 1000
    elif scope_value in {"1", "2mb", "2lb"}:
        threshold = 100
    else:
        return None

    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value) or numeric_value < threshold:
        return None
    return "t CO2e"


def remove_decimal_commas_in_numbers(raw_number: str) -> str:
    """Remove commas in numbers (e.g: 53,452,349 -> 53452349)

    It is silently assumed that the fractional part in any number does not have exactly three digits (i.e.,
    53,452,349 != 53452.349)

    Does the function work correctly with numbers having more than 9 digits?
    """

    pattern1 = r'([0-9]{1,3}),([0-9]{3}),([0-9]{3})'
    pattern2 = r'([0-9]{1,3}),([0-9]{3})'

    if (re.search(pattern1, raw_number) != 'None') | (re.search(pattern2, raw_number) != 'None'):
        res = raw_number.replace(",", "")
    else:
        res = raw_number

    return res


def extract_value_context(value, page_text, window: int = 100) -> str:
    """Return up to ``window`` characters of page text on each side of ``value``.

    Tries a direct substring match first. If that fails, reconstructs the
    likely thousands-separator form (US/UK convention) and retries — this
    covers the common case where ``page_text`` shows ``23,301`` while
    ``value`` has been comma-stripped to ``23301`` upstream. Trailing ``.0``
    on the search key is also tolerated, since pandas float coercion
    introduces it for integer values. Returns an empty string when no match
    is found; whitespace in the returned slice is collapsed to single spaces.
    """
    if value is None or page_text is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(page_text)
    raw = str(value).strip()
    if not raw or not text:
        return ""

    # Pandas float coercion writes integers as ``28714.0`` even when the page
    # said ``28714``. Drop a trailing ``.0`` from the search key so it doesn't
    # poison downstream candidate construction.
    key = raw[:-2] if raw.endswith(".0") else raw

    span = None
    idx = text.find(key)
    if idx != -1:
        span = (idx, idx + len(key))

    if span is None:
        if "." in key:
            int_part, frac_part = key.split(".", 1)
            head = re.sub(r"\D", "", int_part)
            tail = re.sub(r"\D", "", frac_part)
        else:
            head, tail = re.sub(r"\D", "", key), ""
        if len(head) > 3:
            grouped = head[: len(head) % 3 or 3]
            rest = head[len(grouped):]
            grouped += "".join("," + rest[i:i + 3] for i in range(0, len(rest), 3))
            us_candidate = grouped + (("." + tail) if tail else "")
            idx = text.find(us_candidate)
            if idx != -1:
                span = (idx, idx + len(us_candidate))

    if span is None:
        return ""

    start = max(0, span[0] - window)
    end = min(len(text), span[1] + window)
    snippet = text[start:end]
    return re.sub(r"\s+", " ", snippet).strip()


def expand_grid(data_dict):
    """
    Create a dataframe from all combinations of provided lists or arrays.

    This function takes a dictionary of lists or arrays and computes the cartesian product of these lists or arrays.
    Each unique combination of elements will form a row in the resulting dataframe. The keys of the dictionary will
    be used as column names in the dataframe.

    Parameters:
    data_dict (dict): A dictionary where keys are column names and values are lists or arrays containing data.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the cartesian product of the provided lists or arrays.

    Example:
        data_dict = {'height': [60, 70], 'weight': [100, 150, 200]}

        expand_grid(data_dict)

       height  weight
    0      60     100
    1      60     150
    2      60     200
    3      70     100
    4      70     150
    5      70     200
    """
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())


def get_project_directory(path_to_file="src"):
    """
    Problem solved here (poor solution): when using Jupyter Notebooks the working directory is the path of the Jupyter Notebook,
    not the project directory we would like to use.

    See also https://stackoverflow.com/questions/69394705/how-to-set-the-default-working-directory-of-all-jupyter-notebooks-as-the-project

    :param path_to_file: path, where the current file is located
    :return: project directory path
    """

    current_dir = os.getcwd()
    head, tail = os.path.split(current_dir)

    # Is there an automatic way to get the current file name?
    # import ipyparams
    # current_file_name = ipyparams.notebook_name
    # print(current_notebook_name)

    if tail == "src":
        return head
    elif tail == path_to_file:
        return os.path.dirname(os.path.dirname(head))
    else:
        return current_dir


def check_loop():
    """Check if running in Jupyter notebook or not."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # 'RuntimeError: There is no current event loop...'
        loop = None

    if loop and loop.is_running():
        return True
    else:
        return False


def read_txt_file(txt_input_path):
    """Read a text file and return its content as a string."""
    with open(txt_input_path, 'r', encoding='utf-8') as file:
        return file.read()


def read_json_to_str(input_json_path):
    """Read a JSON file and return its content as a string."""

    read_definitions = read_txt_file(input_json_path)

    definitions_dict = json.loads(read_definitions)

    definitions_string = ' '.join(definitions_dict.values())

    return definitions_string


def read_output_files(output_path):
    """Read output table file in specified folder."""

    output_file_path = output_path + 'raw_results.csv'

    # Read the output table
    output_table = pd.read_csv(output_file_path)

    return output_table


# Confidence utility for log-probabilities
def compute_value_confidence(value_str: str, logprobs_content) -> float | None:
    """Return the probability that the LLM assigned to *exactly* ``value_str``.

    The function aggregates per-token log-probabilities that Azure/OpenAI can
    return when the ``logprobs`` parameter is enabled in a chat-completion
    request. It manually calculates the character position of each token to
    find which tokens from the log-probs list correspond to the extracted
    ``value_str``.

    Parameters
    ----------
    value_str : str
        The numeric value extracted from the LLM output (e.g. "14.7").
    logprobs_content : list | None
        A list of `ChatCompletionTokenLogprob` objects.

    Returns
    -------
    float | None
        A probability in the interval (0, 1].  ``None`` if inputs are missing
        or the value substring could not be located.
    """

    if not logprobs_content or not value_str:
        return None

    # Ensure value_str is a string (handles floats from Pydantic)
    value_str = str(value_str) if value_str is not None else ""

    # Handle non-numeric/non-value cases
    if value_str.lower() in ["not specified", "null", "none", "na"]:
        return None

    # Normalize for matching (handles spaces, commas, case)
    clean_value = value_str.strip().replace(" ", "").replace(
        ",", "").lower()  # e.g., " 1,056.1 " → "1056.1"

    # Find relevant logprobs by matching tokens in logprobs_content
    relevant_logprobs = []
    for item in logprobs_content:
        clean_token = item.token.strip().replace(" ", "").replace(",", "").lower()
        if clean_token in clean_value:  # Simple substring match
            relevant_logprobs.append(item.logprob)

    # Handle no matches (e.g., if tokenization doesn't align)
    if not relevant_logprobs:
        return None

    # Filter out punctuation to avoid skewing (e.g., focus on digits)
    relevant_logprobs = [
        lp for i, lp in enumerate(relevant_logprobs)
        if logprobs_content[i].token not in [",", ".", " ", "-"]
    ]
    if not relevant_logprobs:
        return 0.0  # Fallback for no meaningful tokens

    # NOTE: Substring matching over tokens with Structured JSON prompt
    # can over-include unrelated tokens if the same token text (value)
    # appears elsewhere in the completion (LLM output).

    # exp(sum(logprobs))
    joint_log_prob = math.fsum(relevant_logprobs)  # sum up the log-probs
    joint_prob = math.exp(joint_log_prob)  # calculate the joint probability

    return joint_prob


# For Regex-based prompter classes
def compute_substring_probability(match_obj, group_num: int, log_blocks) -> float | None:
    """
    Compute probability for a specific regex match group using log-probabilities.

    This function calculates the joint probability of tokens that correspond to a specific
    capture group in a regex match, using token-level log-probabilities from the LLM.

    Args:
        match_obj: Regex match object containing the matched groups
        group_num: Group number to extract (e.g., 3 for value, 5 for unit)
        log_blocks: List of ChatCompletionTokenLogprob objects from LLM response

    Returns:
        float | None: Joint probability value in (0,1] or None if not computable
    """
    if not log_blocks or not match_obj.group(group_num):
        return None

    tokens = log_blocks  # list of ChatCompletionTokenLogprob

    # Map the matched substring [start, end) in the completion string to token indices.
    # cum_end holds exclusive end char positions for each token in sequence.
    # - start_idx: first token whose end > start (token overlapping the start position)
    # - end_idx: include the token whose end >= end (hence j = bisect_left(...); slice end is j+1)
    lengths = [len(t.token) for t in tokens]
    # [end0, end1, ...], exclusive ends in chars
    cum_end = list(accumulate(lengths))

    start = match_obj.start(group_num)
    end = match_obj.end(group_num)  # exclusive
    start_idx = bisect.bisect_right(cum_end, start)
    j = bisect.bisect_left(cum_end, end)
    end_idx = min(j + 1, len(tokens))
    selected = tokens[start_idx:end_idx]

    if selected:
        joint_log_prob = math.fsum(t.logprob for t in selected)
        return math.exp(joint_log_prob)

    return None


def get_unit_normalization_mapping(df: pd.DataFrame,
                                   unit_col_name: str,
                                   pipeline_output_flag: bool,
                                   unit_normalization_path: str = _DEFAULT_UNIT_NORMALIZATION_PATH) -> pd.DataFrame:
    """Add normalized units using the dictionary first and classifier as fallback."""
    unit_normalization_dict = pd.read_csv(unit_normalization_path)
    # Create a mapping/dictionary from `unit_normalization_df`
    unit_mapping = dict(
        zip(unit_normalization_dict["unit"],
            unit_normalization_dict["normalized_unit"])
    )

    unit_normalized_col_name = unit_col_name + "_normalized"
    unit_normalization_method_col_name = unit_col_name + "_normalization_method"

    def map_unit(row):
        unit = row
        if pd.isna(unit):  # Case 1: If the unit is NaN
            return pd.NA, None, pd.NA
        # Case 2: If no unit was extracted
        elif (unit == 'Nothing extracted. No Regex match') or (unit == 'Not specified'):
            return pd.NA, None, pd.NA
        elif not isinstance(unit, str):
            return "Unknown", None, pd.NA

        stripped_unit = unit.strip()
        if stripped_unit in unit_mapping:
            return unit_mapping[stripped_unit], None, "dictionary"
        if not stripped_unit:
            return "Unknown", None, pd.NA
        return "Unknown", unit, "classifier"

    mapped_units = [map_unit(unit) for unit in df[unit_col_name]]
    unmatched_units = list(dict.fromkeys(
        unit for _, unit, _ in mapped_units if unit is not None
    ))
    predictions = dict(zip(
        unmatched_units, _predict_unit_normalizations(unmatched_units)
    )) if unmatched_units else {}
    df[unit_normalized_col_name] = [
        predictions.get(unmatched, normalized)
        for normalized, unmatched, _ in mapped_units
    ]
    unit_normalization_methods = [
        method for _, _, method in mapped_units
    ]
    if pipeline_output_flag:
        required_columns = {"value", "extracted_scope_from_llm"}
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise ValueError(
                "Pipeline unit correction requires columns: "
                + ", ".join(sorted(missing_columns))
            )
        corrected_units = []
        corrected_methods = []
        for unit, value, scope, normalized, method in zip(
            df[unit_col_name],
            df["value"],
            df["extracted_scope_from_llm"],
            df[unit_normalized_col_name],
            unit_normalization_methods,
        ):
            if isinstance(normalized, str) and normalized == "Mt CO2e":
                corrected = _correct_ambiguous_mt_unit(unit, value, scope)
                if corrected is not None:
                    normalized = corrected
                    method = f"{method} (corrected from Mt CO2e to t CO2e)"
            corrected_units.append(normalized)
            corrected_methods.append(method)
        df[unit_normalized_col_name] = corrected_units
        unit_normalization_methods = corrected_methods
    df[unit_normalization_method_col_name] = unit_normalization_methods

    return df


def get_value_standardization(df: pd.DataFrame,
                              value_column_name: str,
                              unit_column_name: str,
                              unit_normalization_path: str = _DEFAULT_UNIT_NORMALIZATION_PATH) -> pd.DataFrame:
    """Adds a standardized value column to the df"""
    standardized_value_column_name = 'standardized_' + value_column_name
    df[standardized_value_column_name] = pd.NA

    # Read the unit normalization mapping
    unit_normalization_dict = pd.read_csv(unit_normalization_path)

    unit_mapping = dict(
        zip(unit_normalization_dict["normalized_unit"],
            unit_normalization_dict["factor"])
    )

    df['factor'] = df.apply(
        lambda row: unit_mapping[row[unit_column_name]]
        if row[unit_column_name] in unit_mapping else pd.NA,
        axis=1
    )

    # Create mask where 'value' and 'factor' are both present and numeric
    mask = (
        df[value_column_name].notna() &
        df['factor'].notna() &
        pd.to_numeric(df[value_column_name], errors='coerce').notna() &
        pd.to_numeric(df['factor'], errors='coerce').notna()
    )

    # Perform multiplication where mask is True
    df.loc[mask, standardized_value_column_name] = (
        df.loc[mask, value_column_name].astype(
            float) * df.loc[mask, 'factor'].astype(float)
    )

    return df
