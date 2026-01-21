"""
Module to identify and resolve duplicates in the extracted data and the ground truth.
"""
from typing import List

import pandas as pd


def handle_duplicates_in_ground_truth(
    df: pd.DataFrame,
    col_names: List[str],
    select: bool = False,
    preferred_unit: str = "t CO2e"
) -> pd.DataFrame:
    """
    High-level function to identify, mark, and resolve duplicates in the dataframe.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        col_names (List[str]): List of column names. First three columns are used for grouping.
        select (bool): Whether to filter the dataframe to include only resolved duplicates.
        preferred_unit (str): Preferred unit for resolving duplicates (e.g., "t CO2e").

    Returns:
        pd.DataFrame: Duplicate-resolved dataframe.
    """
    # Identify and resolve duplicates in one step
    df = identify_duplicates_in_ground_truth(df, col_names, preferred_unit)

    if select:
        df = select_duplicates_in_ground_truth(df)

    return df


def identify_duplicates_in_ground_truth(
    df: pd.DataFrame, col_names: List[str], preferred_unit: str
) -> pd.DataFrame:
    """
    Identify duplicates and resolve them with prioritization rules.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        col_names (List[str]): Columns for grouping and identifying duplicates.
        preferred_unit (str): Preferred unit for resolving duplicates (e.g., "t CO2e").

    Returns:
        pd.DataFrame: Updated dataframe with duplicate-related flags applied.
    """
    # Extract column names
    report_name_col, scope_col, year_col, _, _, _ = col_names

    # Step 1: Identify duplicates
    grouped_counts = df.groupby([report_name_col, scope_col, year_col]).size()

    # Add duplicate flag
    df["duplicate_flag"] = df.set_index([report_name_col, scope_col, year_col]).index.isin(
        grouped_counts[grouped_counts > 1].index)
    df["duplicate_flag"] = df["duplicate_flag"].fillna(False)

    # Step 2: Filter duplicates for resolution
    df_duplicates = df[df["duplicate_flag"]].copy()

    # Apply prioritization steps
    df_duplicates = _apply_prioritization_rules_for_ground_truth(
        df_duplicates, col_names, preferred_unit)

    # Mark resolved rows as selected in the original dataframe
    df = mark_selected_rows(df, df_duplicates, col_names)

    return df


def _apply_prioritization_rules_for_ground_truth(df: pd.DataFrame,
                                                 col_names: List[str],
                                                 preferred_unit: str) -> pd.DataFrame:
    """
    Resolve duplicate rows using prioritization rules in three steps.
    1. Keep first occurrence of identical entries (identical = same value and unit on same page).
    2. Keep entry with preferred unit.
    3. Keep the entry from the page with the most matches (majority page).

    Parameters:
        df (pd.DataFrame): Duplicate rows to resolve.
        col_names (List[str]): Columns for grouping and identifying duplicates.
        preferred_unit (str): Preferred unit for resolving duplicates (e.g., "t CO2e").

    Returns:
        pd.DataFrame: Resolved duplicate rows.
    """
    report_name_col, scope_col, year_col, page_col, _, unit_norm_col = col_names

    def prefer_unit(group: pd.DataFrame) -> pd.DataFrame:
        """Keep rows with the preferred unit (`preferred_unit`) if available."""
        if preferred_unit in group[unit_norm_col].values:
            return group[group[unit_norm_col] == preferred_unit]
        return group

    def majority_page(group: pd.DataFrame) -> pd.DataFrame:
        """Keep rows with the most frequent page (`mode`)."""
        mode_page = group[page_col].mode()
        if not mode_page.empty:
            return group[group[page_col] == mode_page.iloc[0]]
        return group

    # Step 1: Drop identical entries
    df = df.drop_duplicates(subset=col_names, keep="first")

    # Step 2: Preferred unit
    df = df.groupby([report_name_col, scope_col, year_col]).apply(
        prefer_unit).reset_index(drop=True)

    # Step 3: Majority page
    df = df.groupby([report_name_col, scope_col, year_col]).apply(
        majority_page).reset_index(drop=True)

    return df


def mark_selected_rows(
    original_df: pd.DataFrame, resolved_df: pd.DataFrame, col_names: List[str]
) -> pd.DataFrame:
    """
    Mark rows in the original dataframe based on resolved duplicates.

    Parameters:
        original_df (pd.DataFrame): Original input dataframe.
        resolved_df (pd.DataFrame): Duplicate rows resolved (selected rows).
        col_names (List[str]): Columns for joining the original and resolved dataframes.

    Returns:
        pd.DataFrame: Original dataframe with `select_flag` updated.
    """
    # Add select_flag column from resolved duplicates
    resolved_df["select_flag"] = True
    updated_df = original_df.merge(
        resolved_df[[*col_names, "select_flag"]],
        on=col_names,
        how="left"
    )

    # Fill NA in 'select_flag' for duplicates that were not resolved (set False)
    updated_df.loc[updated_df['duplicate_flag'],
                   'select_flag'] = updated_df['select_flag'].fillna(False)

    return updated_df


def handle_duplicates_in_output(df: pd.DataFrame,
                                col_names: List[str],
                                select: bool = False,
                                preferred_unit: str = "t CO2e") -> pd.DataFrame:
    """Identify and resolve duplicates in the output dataframe
    based on specified columns."""
    df = identify_duplicates_in_output(
        df, col_names, preferred_unit)

    if select:
        df = select_duplicates_in_output(df)

    return df


def identify_duplicates_in_output(df: pd.DataFrame,
                                  col_names: List[str],
                                  preferred_unit: str) -> pd.DataFrame:
    """
    High-level function to identify duplicates in the output dataframe and 
    apply rules for deduplication.
    """
    report_col, scope_col, year_col, _, value_col, _ = col_names

    # Step 1: Mark all-NA reports
    df["all_na"] = df.groupby(report_col)[value_col].transform(
        lambda x: x.isna().all())

    # Step 2: Handle all-NA reports separately
    all_na_reports = _filter_all_na_reports(
        df, "all_na", [report_col, scope_col, year_col])

    # Step 3: Focus on reports with non-NA values
    reports_with_non_na_values = df[~df["all_na"]].copy()

    # Step 4: Filter rows with non-NA values
    non_na_rows = reports_with_non_na_values[reports_with_non_na_values[value_col].notna(
    )].copy()

    # Step 5: Identify unique and non-unique rows
    duplicate_counts = non_na_rows.groupby(
        [report_col, scope_col, year_col]).size()
    unique_combinations = duplicate_counts[duplicate_counts == 1].index
    non_unique_combinations = duplicate_counts[duplicate_counts > 1].index

    # Step 6: Extract unique rows
    unique_rows = non_na_rows[
        non_na_rows.set_index([report_col, scope_col, year_col]).index.isin(
            unique_combinations)
    ].copy()

    # Step 7: Extract and prioritize non-unique rows
    non_unique_rows = non_na_rows[
        non_na_rows.set_index([report_col, scope_col, year_col]).index.isin(
            non_unique_combinations)
    ].copy()
    if not non_unique_rows.empty:
        non_unique_rows["duplicate_flag"] = True
        prioritized_non_unique_rows = _apply_prioritization_rules_for_output(
            non_unique_rows, col_names, preferred_unit)

        # Mark selected rows
        prioritized_non_unique_rows["select_flag"] = True
        marked_non_unique_rows = non_unique_rows.merge(
            prioritized_non_unique_rows[[
                *col_names, "select_flag", "dupl_reason"]],
            on=col_names,
            how="left"
        )
        marked_non_unique_rows["select_flag"] = marked_non_unique_rows["select_flag"].fillna(
            False)
        # Fill dupl_reason for non-selected rows with 0
        marked_non_unique_rows["dupl_reason"] = marked_non_unique_rows["dupl_reason"].fillna(
            0)

    else:
        marked_non_unique_rows = pd.DataFrame(
            columns=non_na_rows.columns.tolist() +
            ["duplicate_flag"] + ["select_flag"] + ["dupl_reason"])

    # Step 8: Handle NA rows
    na_rows = _deduplicate_na_rows(reports_with_non_na_values, [
                                   report_col, scope_col, year_col], value_col)

    # Step 9: Combine subsets (unique rows, marked non-unique rows, and NA rows)
    combined_rows = _combine_subsets(
        unique_rows, marked_non_unique_rows, na_rows, [report_col, scope_col, year_col])

    # Step 10: Add back all-NA reports
    df_filtered = pd.concat([combined_rows, all_na_reports])
    df_filtered["duplicate_flag"] = df_filtered["duplicate_flag"].fillna(False)

    # Mark filtered rows in original dataframe
    # Drop unnecessary columns before merging
    df_merge = df_filtered.drop(columns=["text_response_from_llm",
                                         "page_numbers_tried_by_llm"], axis=1)
    col_for_merge = [col for col in df.columns if col not in ["text_response_from_llm",
                                                              "page_numbers_tried_by_llm"]]
    df = df.merge(df_merge, how="left", on=col_for_merge, indicator=True)

    return df


def select_duplicates_in_output(df: pd.DataFrame) -> pd.DataFrame:
    """Select non-duplicates and selected duplicates."""
    df = df.loc[df['_merge'] == 'both'].copy().drop(columns=['_merge'])
    df_resolved = df.loc[
        (df['duplicate_flag'] == False) | (
            df['select_flag'] == True)
    ].copy()
    return df_resolved

# Private utility functions


def _filter_all_na_reports(df: pd.DataFrame, col_name: str,
                           group_cols: List[str]) -> pd.DataFrame:
    """
    Keep first occurrence of all-NA reports.
    """
    all_na_reports = df[df[col_name]].copy()
    all_na_reports = all_na_reports.drop_duplicates(
        subset=group_cols, keep="first")
    return all_na_reports


def _apply_prioritization_rules_for_output(df: pd.DataFrame,
                                           col_names: List[str],
                                           preferred_unit: str) -> pd.DataFrame:
    """
    Apply prioritization rules to deduplicate entries:
    1. Keep first occurrence of identical entries (identical = same value and unit on same page).
    2. Keep entry with preferred unit.
    3. Keep the entry of the page from the majority page
    """
    group_cols, unit_col = col_names[:3], col_names[-1]
    df = df.copy()
    df['dupl_reason'] = None
    # Rule 1: Deduplicate identical entries (same value and unit on same page)
    initial_len = len(df)
    df_filtered = df.drop_duplicates(
        subset=col_names, keep='first')
    if len(df_filtered) < initial_len:
        df_filtered['dupl_reason'] = 1

    def _filter_by_preferred_unit(group: pd.DataFrame, preferred_unit: str) -> pd.DataFrame:
        """
        Filter rows in a group, keeping only rows with the preferred unit.
        """
        before = len(group)
        if preferred_unit in group[unit_col].dropna().values:
            filtered = group[group[unit_col] == preferred_unit].copy()
            if len(filtered) < before:
                filtered['dupl_reason'] = 2
            return filtered
        return group

    def _filter_by_majority_page(group: pd.DataFrame) -> pd.DataFrame:
        """
        Keep rows with the most frequent page (`mode`).
        """
        before = len(group)
        mode_page = group['page_number_used_by_llm'].mode()
        if not mode_page.empty:
            filtered = group[group['page_number_used_by_llm']
                             == mode_page.iloc[0]].copy()
            if len(filtered) < before:
                filtered['dupl_reason'] = 3
            return filtered
        return group

    # Rule 2: Keep enty with preferred unit
    preferred_unit_filtered = df_filtered.groupby(group_cols, group_keys=False).apply(
        lambda g: _filter_by_preferred_unit(g, preferred_unit)).reset_index(drop=True)

    # Rule 3: Keep entry from page with majority page
    prioritized_rows = preferred_unit_filtered.groupby(group_cols, group_keys=False).apply(
        _filter_by_majority_page).reset_index(drop=True)

    # Check if any group has more than one row left
    duplicates_check = prioritized_rows.groupby(group_cols).size()
    if any(duplicates_check > 1):
        print("Warning: Some groups still have multiple rows after prioritization.")

    return prioritized_rows


def _deduplicate_na_rows(df: pd.DataFrame, group_cols: List[str], value_col: str) -> pd.DataFrame:
    """
    Handle rows where the value is NA by deduplicating them on group columns.
    """
    na_rows = df[df[value_col].isna()].copy()
    return na_rows.drop_duplicates(subset=group_cols, keep="first")


def _combine_subsets(unique_rows: pd.DataFrame,
                     marked_non_unique_rows: pd.DataFrame,
                     na_rows: pd.DataFrame,
                     group_cols: List[str]) -> pd.DataFrame:
    """
    Combine unique rows, marked non-unique rows, and NA rows into a single DataFrame.
    """
    # Only add those NA rows which are not already present in df_non_na_subset
    # (So we do not get one scope-year combination with NA and the other one with a value)
    non_na_subset = pd.concat([unique_rows, marked_non_unique_rows])
    na_rows = na_rows[~na_rows.set_index(
        group_cols).index.isin(
            non_na_subset.set_index(
                group_cols).index)]
    return pd.concat([non_na_subset, na_rows])


def select_duplicates_in_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    """Select non-duplicates and selected duplicates."""
    df_resolved = df.loc[
        (~df['duplicate_flag']) | (
            df['select_flag'])
    ].copy()
    return df_resolved
