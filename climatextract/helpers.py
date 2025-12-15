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
from itertools import accumulate

import numpy as np
import pandas as pd


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

    output_file_path = output_path + '03_co2_emission_table2_w_query_responses.csv'

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

    # NOTE: Substring matching over tokens with Custom GAIA
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
                                   unit_normalization_path: str = "data/normalization_units/unit_normalization_dict.csv") -> pd.DataFrame:
    """Adds a normalized unit column to the df"""
    unit_normalization_dict = pd.read_csv(unit_normalization_path)
    # Create a mapping/dictionary from `unit_normalization_df`
    unit_mapping = dict(
        zip(unit_normalization_dict["unit"],
            unit_normalization_dict["normalized_unit"])
    )

    # Find unit in dict and set normalized value according to mapping
    unit_normalized_col_name = unit_col_name + "_normalized"

    def map_unit(row):
        unit = row[unit_col_name]
        if pd.isna(unit):  # Case 1: If the unit is NaN
            return pd.NA
        # Case 2: If no unit was extracted
        elif (unit == 'Nothing extracted. No Regex match') or (unit == 'Not specified'):
            return pd.NA
        elif unit.strip() not in unit_mapping:  # Case 2: If the unit is not in the dictionary
            return "Unknown"
        else:  # Case 3: If the unit exists in the dictionary
            return unit_mapping[unit.strip()]

    # Apply the `map_unit` function row-wise
    df[unit_normalized_col_name] = df.apply(map_unit, axis=1)

    return df


def get_value_standardization(df: pd.DataFrame,
                              value_column_name: str,
                              unit_column_name: str,
                              unit_normalization_path: str = "data/normalization_units/unit_normalization_dict.csv") -> pd.DataFrame:
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
