"""Helper functions for evaluation of our pipeline"""

from typing import List
import pandas as pd


def compare_value_columns(row):
    """Compares the values of two columns in a DataFrame row: 
    'value_man' and 'extracted_value_from_llm'.

    Args:
        row (pd.Series): A row of a DataFrame containing the columns 
                        'value_man' and 'extracted_value_from_llm'.

    Returns:
        bool: True if both values are NaN or if they are equal; otherwise, False.
    """
    if pd.isna(row['value_man']) and pd.isna(row['extracted_value_from_llm']):
        return None
    else:
        return row['value_man'] == row['extracted_value_from_llm']


def compare_unit_columns(row):
    """Compares the values of two columns in a DataFrame row: 
    'unit_man' and 'extracted_unit_from_llm'.

    Parameters:
        row (pd.Series): A row of a DataFrame containing the columns 
                        'unit_man' and 'extracted_unit_from_llm'.

    Returns:
        bool: True if 'unit_man' is NaN and 'extracted_unit_from_llm' is 
            'Not specified' or 'Nothing extracted. No Regex match'; otherwise, False.
    """

    if pd.isna(row['unit_man']) and pd.isna(row['extracted_unit_from_llm']):
        return None
    else:
        extracted_unit = row['extracted_unit_from_llm'].strip() if not pd.isna(
            row['extracted_unit_from_llm']) else row['extracted_unit_from_llm']
        return row['unit_man'] == extracted_unit


def compare_unit_normalized_columns(row):
    """Compares the values of two columns in a DataFrame row: 
    'unit_man' and 'extracted_unit_from_llm'.

    Parameters:
        row (pd.Series): A row of a DataFrame containing the columns 
                        'unit_man_normalized' and 'unit_normalized'.

    Returns:
        bool: True if 'unit_man_normalized' is NaN and 'unit_normalized' is 
            'Not specified' or 'Nothing extracted. No Regex match'; otherwise, False.
    """

    if pd.isna(row['unit_man_normalized']) and pd.isna(row['normalized_unit_from_dictionary']):
        return None
    else:
        extracted_unit = row['normalized_unit_from_dictionary'].strip() if not pd.isna(
            row['normalized_unit_from_dictionary']) else row['normalized_unit_from_dictionary']
        return row['unit_man_normalized'] == extracted_unit


def bitwise_and_with_none(a: bool | None, b: bool | None) -> bool:
    """Bitwise AND operation with None."""
    if a is None or b is None:
        return None
    else:
        return a & b


def page_in_range(page: str, page_range: List[str] | None) -> bool:
    """Check if a page number is in a given range."""
    if not isinstance(page_range, list) or page_range is None:
        return None
    if pd.isna(page):
        return None
    elif page in page_range:
        return True
    else:
        return False


def compare_page_columns(row: pd.Series) -> bool:
    """Compare the page number of the ground truth and the results."""
    if pd.isna(row['page_number_used_by_llm']):  # and pd.isna(row['page_man'])
        return None
    else:
        return row['page_man'] == row['page_number_used_by_llm']


def define_operations() -> dict[str, List]:
    """Define type of aggregation operation for each comparison column."""
    return {
        'page_man': ["nunique", "count"],
        'page_suggestion_match': [any_match, "sum", complete_match],
        'page_number_used_by_llm': ["nunique", "count"],
        'page_match': [any_match, "sum", complete_match],
        'value_man': ["nunique", "count"],
        'extracted_value_from_llm': ["nunique", "count"],
        'value_match': [any_match, "sum", complete_match],
        'value_and_unit_match': ["sum", complete_match],
        'value_and_unit_normalized_match': ["sum", complete_match],
        'ms_comment_man': ["unique"],
    }


def any_match(series: pd.Series) -> bool:
    """Check whether there is any True value in column."""
    return series.dropna().any()


def complete_match(series: pd.Series) -> bool:
    """Check wether all values in column are True after dropping NA."""
    return series.dropna().all()


def custom_switch(row: pd.Series) -> str:
    """Set error type depending on rowwise comparison of LLM output and human annotations."""
    if not row[('page_suggestion_match', 'complete_match')]:
        return 'l_1 Retrieval failure - Incomplete text passed to LLM'
    elif (row[('value_man', 'count')] == 0) & (row[('extracted_value_from_llm', 'count')] == 0):
        return 'l_6 correct result - No CO2 emissions found'
    elif (row[('value_man', 'count')] == 0) & (row[('extracted_value_from_llm', 'count')] > 0):
        # paper: LLM extracts information from wrong page
        return 'l_2_1 LLM error - LLM extracts >= 1 wrong values, truth - no ground truth available'
    elif (row[('extracted_value_from_llm', 'count')] > 0) & (not row[('page_match', 'any_match')]):
        # paper: LLM extracts information from wrong page
        return 'l_2_2 LLM error - LLM extracts >= 1 wrong values, truth - ground truth on different page'
    elif not row[('value_match', 'any_match')]:
        return 'l_3 LLM error - LLM fails to find ANY non-NA correct values'
    elif not row[('value_match', 'complete_match')]:
        return 'l_4 LLM error - LLM fails to find ALL non-NA correct values'
    elif not row[('value_and_unit_normalized_match', 'complete_match')]:
        return 'l_5 correct result - Correct values but wrong units extracted'
    elif row[('value_and_unit_normalized_match', 'complete_match')]:
        return 'l_6 correct result - All CO2 emissions extracted'
    else:
        return 'This cannot happen.'


def classify_error(row: pd.Series, on: str) -> str:
    """Classify errors based on the rowwise comparison of LLM output and \
        human annotations for the given variable."""
    if pd.isna(row['value_man']) and pd.isna(row['extracted_value_from_llm']):
        return f'true_negative_{on}'
    elif not pd.isna(row['value_man']) and pd.isna(row['extracted_value_from_llm']):
        return f'false_negative_{on}'
    elif row['value_man'] == row['extracted_value_from_llm']:
        return f'true_positive_{on}'
    elif row['value_man'] != row['extracted_value_from_llm']:
        return f'false_positive_{on}'
    else:
        return 'This cannot happen.'


def compute_precision(row: pd.Series, on: str) -> float:
    """
    Calculate the precision metric for a given row of data.
    Precision is defined as the ratio of true positives 
    to the sum of true positives and false positives.
    It measures the accuracy of the positive predictions.
    Args:
        row (dict): A dictionary containing the keys 'true positive' and 'false positive', 
                    representing the counts of true positive and false positive predictions 
                    respectively.
    Returns:
        float: The precision value. If the sum of true positives and false positives is zero, 
               the function returns 0 to avoid division by zero.
    """

    if row[f'true_positive_{on}'] + row[f'false_positive_{on}'] == 0:
        return 0
    else:
        return row[f'true_positive_{on}'] / (row[f'true_positive_{on}'] + row[f'false_positive_{on}'])


def compute_recall(row: pd.Series, on: str) -> float:
    """
    Compute the recall metric for a given row of data.
    Recall is calculated as the ratio of true positives 
    to the sum of true positives and false negatives.
    It measures the ability of the model to correctly identify positive instances.

    Args:
        row (dict): A dictionary containing 'true positive' and 'false negative' keys 
                    with their respective counts.
    Returns:
        float: The recall value. Returns 0 if the sum of true positives and false negatives is 0.
    """
    if row[f'true_positive_{on}'] + row[f'false_negative_{on}'] == 0:
        return 0
    else:
        return row[f'true_positive_{on}'] / (row[f'true_positive_{on}'] + row[f'false_negative_{on}'])


def compute_f1(row: pd.Series, on: str) -> float:
    """
    Compute the F1 score for a given row of data.
    The F1 score is the harmonic mean of precision and recall, providing a 
    balance between the two metrics. It is particularly useful when you need 
    to take both false positives and false negatives into account.
    Args:
        row (dict or pandas.Series): A data structure containing the necessary 
                                     information to compute precision and recall.
    Returns:
        float: The F1 score, which ranges from 0 to 1. If both precision and 
               recall are zero, the function returns 0.
    """

    precision = compute_precision(row, on)
    recall = compute_recall(row, on)
    if precision + recall == 0:
        return 0
    else:
        return 2 * ((precision * recall) / (precision + recall))
