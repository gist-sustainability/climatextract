"""Module to evaluate our pipeline"""

import os
from typing import Dict, List, Optional
import ast
import pandas as pd

import climatextract.evaluate_helpers as evaluate_helpers
from climatextract.helpers import get_unit_normalization_mapping, get_value_standardization
from climatextract.resolve_duplicates import handle_duplicates_in_ground_truth


class EvaluatorData:
    """Class to instantiate data objects for evaluation."""

    def __init__(self, path_to_results: str, path_to_ground_truth: str):
        """Load the results and ground truth data for comparison."""

        self.ground_truth = self._read_ground_truth(path_to_ground_truth)
        self.path_to_results = path_to_results

        self.results = pd.read_csv(os.path.join(
            self.path_to_results, "03_co2_emission_table2_w_query_responses_filtered.csv"),
            dtype={
                'extracted_scope_from_llm_orig': str,
                'extracted_scope_from_llm': str,
                'page_number_used_by_llm': str,
                'page_number_to_llm': str
        })
        self.results['page_numbers_tried_by_llm'] = self.results['page_numbers_tried_by_llm'].apply(
            ast.literal_eval)

        self.small_results = None
        self.small_results_subset = None

    def _read_ground_truth(self, path_to_ground_truth: str) -> pd.DataFrame:
        if "gist_2025.csv" not in path_to_ground_truth:
            raise ValueError(
                f"Unsupported ground truth source '{path_to_ground_truth}'. "
            )
        ground_truth = self._read_ground_truth_gist_2025(path_to_ground_truth)

        # Change nan to None to align with string type
        ground_truth['page_man'] = ground_truth['page_man'].where(
            pd.notna(ground_truth['page_man']), None)

        # Convert value_man to numeric
        ground_truth['value_man'] = pd.to_numeric(
            ground_truth['value_man'].str.replace(',', '.'), errors='coerce')
        ground_truth['year_man'] = pd.to_numeric(
            ground_truth['year_man'], errors='coerce')
        return ground_truth

    def _read_ground_truth_gist_2025(self, path_to_ground_truth: str) -> pd.DataFrame:
        ground_truth = pd.read_csv(path_to_ground_truth, dtype={
            'scope': str, 'page': str}, decimal=',')
        ground_truth_subset = ground_truth[[
            'report_name', 'scope', 'year',
            'value', 'unit', 'unit_normalized',
            'page', 'metric_name', 'display_type']]
        # TODO: Find alternative to keeping columns metric_name, display_type and
        # ms_comment_man for line 196
        ground_truth_subset = ground_truth_subset.assign(
            ms_comment_man=None)
        rename_columns_map = {
            'report_name': 'ReportName',
            'scope': 'scope_man',
            'year': 'year_man',
            'value': 'value_man',
            'unit': 'unit_man',
            'unit_normalized': 'unit_man_normalized',
            'page': 'page_man',
            'metric_name': 'val_name_man',
            'display_type': 'type_man',
        }
        ground_truth_subset = ground_truth_subset.rename(
            columns=rename_columns_map)

        # Normalize units
        ground_truth_subset = get_unit_normalization_mapping(
            ground_truth_subset, "unit_man", pipeline_output_flag=False)

        # Standardize values
        ground_truth_subset = get_value_standardization(
            ground_truth_subset, "value_man", "unit_man_normalized")

        # Handle duplicates
        col_names = ['ReportName', 'scope_man', 'year_man',
                     'page_man', 'value_man', 'unit_man_normalized']
        ground_truth_subset = handle_duplicates_in_ground_truth(
            ground_truth_subset, col_names, select=True)

        # Drop flags column created in handle_duplicates
        ground_truth_subset = ground_truth_subset.drop(
            columns=['duplicate_flag', 'select_flag'])

        # Rename column "factor" to "factor_man" to avoid confusion
        ground_truth_subset = ground_truth_subset.rename(
            columns={"factor": "factor_man"})

        return ground_truth_subset

    def _merge_data_for_comparison(self) -> pd.DataFrame:
        """
            1. Copy "automatic_extraction_tried",  "page_numbers_tried_by_llm" 
                into ground_truth dataset: grtruth_extended
            2. Remove all years/scopes from LLM output dataset if they are missing: tinyresults
            3. Left-Merge tinyresults into grtruth_extended, 
                to only keep scope-year-combinations from grid
                =(we keep >= 1 row for each scope-year-combination from ground truth) & \
                & (we keep 0-\\infty rows/extracted values from each page)
        """

        # create additional variables "automatic_extraction_tried" and
        # "page_numbers_tried_by_llm" in ground_truth
        grtruth_extended = self.results[[
            "report_name_short", "automatic_extraction_tried",  "page_numbers_tried_by_llm"]]
        # Need to convert list to tuple to be hashable for groupby
        grtruth_extended['page_numbers_tried_by_llm'] = grtruth_extended[
            'page_numbers_tried_by_llm'].apply(
            tuple)
        grtruth_extended = grtruth_extended.groupby([
            "report_name_short", "page_numbers_tried_by_llm"])[
            'automatic_extraction_tried'].apply(lambda group: group.any(skipna=True)).reset_index()
        grtruth_extended['page_numbers_tried_by_llm'] = grtruth_extended[
            'page_numbers_tried_by_llm'].apply(
            list)

        grtruth_extended = pd.merge(self.ground_truth, grtruth_extended, how="left",
                                    left_on="ReportName", right_on="report_name_short")
        grtruth_extended = grtruth_extended.drop('report_name_short', axis=1)
        grtruth_extended.loc[grtruth_extended['automatic_extraction_tried'].isna(
        ), 'automatic_extraction_tried'] = False

        # 2. Strip irrelevant rows from LLM output
        # it is not obvious how to keep rows that have "helpful" data/ extracted values.
        # at this step we lose information about page numbers passed to the LLM
        # if no valuable information was returned.
        #  Option 1:
        tiny_results = self.results[self.results['extracted_value_from_llm'].notnull(
        )]

        # Option 2:
        # tiny_results = co2_emission_table2_w_query_responses[~(
        #            (co2_emission_table2_w_query_responses[
        #                   'extracted_value_from_llm_orig'] == "Not specified") |
        #            (co2_emission_table2_w_query_responses[
        #                  'extracted_value_from_llm_orig'] == "Nothing extracted. No Regex match")
        # )]

        # 3. Left-merge
        tiny_results = tiny_results.drop(
            'automatic_extraction_tried', axis=1)
        tiny_results = tiny_results.drop(
            'page_numbers_tried_by_llm', axis=1)
        merged_results = pd.merge(grtruth_extended, tiny_results, how="left",
                                  left_on=["ReportName",
                                           "scope_man", "year_man"],
                                  right_on=["report_name_short", "extracted_scope_from_llm",
                                            "extracted_year_from_llm"])

        return merged_results


class EvaluatorDefault(EvaluatorData):
    """Class to perform evaluation of RAG pipeline using default evaluation metrics"""

    def run(self):
        """Main evaluation routine that calls all intermediate steps."""
        self.small_results = self._merge_data_for_comparison()
        self._compare_data()
        self._perform_and_print_checks()
        self._save_comparison_reports()
        results_per_doc = self._aggregate_and_save_comparisons(
            aggregate_by=['ReportName'])
        metrics = self._prepare_results_for_mlflow(results_per_doc)

        return metrics

    def _compare_data(self):
        """Compute matches between llm output and ground truth."""
        self.small_results['human_found_co2_emissions'] = ~self.small_results.groupby(
            'ReportName')['value_man'].transform(lambda x: x.isna().all())

        self.small_results['value_match'] = self.small_results.apply(
            evaluate_helpers.compare_value_columns, axis=1)
        self.small_results['unit_match'] = self.small_results.apply(
            evaluate_helpers.compare_unit_columns, axis=1)
        self.small_results['unit_normalized_match'] = self.small_results.apply(
            evaluate_helpers.compare_unit_normalized_columns, axis=1)

        self.small_results["value_and_unit_match"] = self.small_results.agg(
            lambda x: evaluate_helpers.bitwise_and_with_none(
                x['value_match'], x['unit_match']),
            axis=1)
        self.small_results["value_and_unit_normalized_match"] = self.small_results.agg(
            lambda x: evaluate_helpers.bitwise_and_with_none(
                x['value_match'], x['unit_normalized_match']),
            axis=1)
        self.small_results['page_suggestion_match'] = self.small_results.agg(
            lambda x: evaluate_helpers.page_in_range(
                x['page_man'], x['page_numbers_tried_by_llm']), axis=1)
        self.small_results['page_match'] = self.small_results.apply(
            evaluate_helpers.compare_page_columns, axis=1)

        selected_columns = [
            'ReportName',
            # document-level information
            'automatic_extraction_tried', 'human_found_co2_emissions',
            # key from left merge, but there can be multiple rows
            # if LLM found values on different pages !!
            'scope_man', 'year_man',
            'page_numbers_tried_by_llm', 'page_number_used_by_llm', 'page_retrieval_scores',
            'page_man', 'page_suggestion_match', 'page_match',  # do page numbers match?
            # do values match?
            'extracted_value_from_llm_orig', 'extracted_value_from_llm', 'value_man',
            'value_match',
            # do units match?
            'extracted_unit_from_llm', 'unit_man', 'unit_match',
            # do normalized units match?
            'normalized_unit_from_dictionary', 'unit_man_normalized', 'unit_normalized_match',
            # do values and units (normalized) match?
            'value_and_unit_match', 'value_and_unit_normalized_match',
            # other info from manual annotation
            'val_name_man', 'type_man', 'ms_comment_man',
            'extracted_scope_from_llm_orig', 'extracted_scope_from_llm',  # same as 'scope_man'
            'extracted_year_from_llm_orig', 'extracted_year_from_llm',  # same as 'year_man'
            'page_texts_to_llm', 'text_response_from_llm',  # llm input & output
            'report_name'
        ]

        # Conditionally add value_probability if it exists in the DataFrame
        if 'value_probability' in self.small_results.columns:
            # Insert after value_match for logical order
            selected_columns.insert(selected_columns.index(
                'extracted_value_from_llm') + 1, 'value_probability')

        # Conditionally add unit_probability if it exists in the DataFrame
        if 'unit_probability' in self.small_results.columns:
            # Insert after the extracted unit column for logical order
            selected_columns.insert(selected_columns.index(
                'extracted_unit_from_llm') + 1, 'unit_probability')

        # Now select
        self.small_results = self.small_results[selected_columns]

        # self.small_results.to_excel(os.path.join(path_to_analysis, "results_zwischenstand.xlsx"))

    def _perform_and_print_checks(self):
        # TODO: save checks to txt file instead just console
        """Perform basic data checks and print results."""
        print("Please check that automatic extraction was tried with all reports \
            (we will discard other reports now)")
        print(self.small_results.groupby('ReportName')['automatic_extraction_tried'].apply(
            lambda x: x.all()).value_counts())

        # Keep reports only if we actually tried to extract information automatically
        results_subset = self._subset_reports()

        print("Count for how many reports a human annotator found CO2 emissions \
            in the report (both groups will be analyzed separately)")
        print(results_subset.groupby('ReportName')[
            'human_found_co2_emissions'].apply(lambda x: x.all()).value_counts())

        num_reports = results_subset['ReportName'].nunique()
        # TODO: update this check to 4 scopes and 10 years
        print("Size of universe (expected): " + str(3 * 16 * num_reports) +
              " (3 scopes * 16 years * " + str(num_reports) + " reports)")
        print("Size of universe (manual): " + str(len(self.ground_truth)))
        print("We would expect just one value(by construction) for each \
            (Report, Scope, Year)-combination, but humans found sometimes more than one: ")
        temp = self.ground_truth.groupby(
            ['ReportName', 'scope_man', 'year_man']).size()
        print(temp[temp > 1])
        print("Size of universe (manual + automatic): " +
              str(len(self.small_results)))
        print("If more than one value gets extracted automatically for for any  \
            (Report, Scope, Year)-combination, then (manual + automatic) > manual. \
            It is quite unclear how this would influence self.small_results below and \
            would need to be checked carefully.")

        # By construction, we should have 3 * 16 = 48 rows for each report. If we have more rows,
        # multiple values were extracted from different pages
        # but referring to the same scope and year.
        # (If we believe that reports mention the desired value a single time,
        # this indicates extraction errors!)
        self.small_results.groupby('ReportName')[
            'automatic_extraction_tried'].value_counts()

    def _save_comparison_reports(self):
        """Save detailed comparisons between human & automated annotations \
            separating them in two groups: information is or is not in sustainability report."""
        results_subset = self._subset_reports()
        results_available_in_report = results_subset[results_subset["human_found_co2_emissions"]]
        results_not_available_in_report = results_subset[
            ~results_subset["human_found_co2_emissions"]]

        results_available_in_report.to_csv(os.path.join(
            self.path_to_results, "04a_results_available_in_report.csv"))
        results_not_available_in_report.to_csv(os.path.join(
            self.path_to_results, "04b_results_not_available_in_report.csv"))

        self.small_results_subset = results_subset

    def _subset_reports(self) -> pd.DataFrame:
        """Keep reports only if we actually tried to extract information automatically."""
        return self.small_results[self.small_results["automatic_extraction_tried"]]

    def _aggregate_and_save_comparisons(self, aggregate_by: List[str]):
        """Aggregate comparison results by variables listed in aggregate_by and /
        save aggregated results to path_to_results."""
        operations = evaluate_helpers.define_operations()

        # if len(aggregate_by) == 1:
        # aggregate_by = aggregate_by[0]

        # one record for each aggregation level
        grouped_df = self.small_results_subset.groupby(
            aggregate_by).agg(operations)

        grouped_df[('process_match', 'summary')] = grouped_df.apply(
            evaluate_helpers.custom_switch, axis=1)
        # move last column to the beginning
        column_tuples = grouped_df.columns.tolist()
        column_tuples.insert(0, column_tuples.pop())
        grouped_df = grouped_df[column_tuples]

        grouped_df[('emissions_value_man', 'available_in_report')
                   ] = grouped_df[('value_man', 'nunique')] > 0
        # move last column to the beginning
        column_tuples = grouped_df.columns.tolist()
        column_tuples.insert(0, column_tuples.pop())
        grouped_df = grouped_df[column_tuples]

        # save aggregated results
        file_suffix = "_and_".join(
            aggregate_by) if len(aggregate_by) > 1 else "".join(aggregate_by)
        grouped_df.to_csv(os.path.join(
            self.path_to_results, f"05_results_aggregated_by_{file_suffix}.csv"))

        return grouped_df

    def _prepare_results_for_mlflow(self, results_per_doc):
        """Prepare dictionary for mlflow logging."""

        metrics = {}

        column_pairs = [
            ('process_match', 'summary'),
            ('value_and_unit_match', 'sum'),
            ('value_match', 'sum'),
            ('page_match', 'sum'),
            ('value_and_unit_match', 'complete_match'),
            ('value_and_unit_normalized_match', 'complete_match'),
            ('value_match', 'complete_match'),
            ('page_match', 'complete_match')
        ]

        # Compute value counts for each pair
        for col1, col2 in column_pairs:
            if col2 == 'summary':
                summary_counts = results_per_doc[col1,
                                                 col2].value_counts().to_dict()
                for summary_key, count in summary_counts.items():
                    metrics[f"{summary_key}"] = count

            elif col2 == 'complete_match' or col2 == 'sum':
                metrics[f"{col1}_{col2}"] = results_per_doc[col1, col2].sum()

        self._sanity_check(metrics)
        return metrics

    def _sanity_check(self, metrics: Dict[str, int]):
        """Check if total number of errors is equal to number of reports
        where automatic extraction was tried"""
        total_errors = 0
        for key, value in metrics.items():
            if key.startswith('l_'):
                total_errors += value
        total_reports = self.results['report_name_short'].nunique()
        if total_errors != total_reports:
            raise ValueError(
                "Total number of errors does not match number of reports "
                "where automatic extraction was tried")
        return


class EvaluatorPrecisionRecallF1(EvaluatorData):
    """Class to perform evaluation of RAG pipeline using precision, recall, and F1 score."""

    def run(self):
        """Evaluate the precision, recall, and F1 score of the extracted data."""
        merged_data = self._merge_data_for_comparison()
        results = self._classify_errors(merged_data)
        results_overall = self._compute_metrics_overall(results)
        results_per_doc = self._compute_metrics_per_doc(results)

        self._save_results(
            results_per_doc, self.path_to_results, 'error_analysis_per_doc.csv')
        self._save_results(results, self.path_to_results,
                           'error_analysis_per_row.csv')

        return results_overall

    def _classify_errors(self, merged_data):
        """Compare the results and ground truth data sets."""
        # For now only comparison on value
        # Later: extend to unit and value AND unit
        eval_data = merged_data[merged_data['automatic_extraction_tried']]
        eval_data['error_value'] = eval_data.apply(
            evaluate_helpers.classify_error, axis=1, on="value")

        return eval_data

    def _compute_metrics_overall(self, merged_data, on="value"):
        """Compute overall metrics."""
        all_errors = [f'true_negative_{on}', f'false_positive_{on}',
                      f'true_positive_{on}', f'false_negative_{on}']
        values_overall = merged_data['error_value'].value_counts().reindex(
            all_errors, fill_value=0)

        self._sanity_check_overall(values_overall)

        values_overall['precision_value'] = evaluate_helpers.compute_precision(
            values_overall, on)
        values_overall['recall_value'] = evaluate_helpers.compute_recall(
            values_overall, on)
        values_overall['f1_value'] = evaluate_helpers.compute_f1(
            values_overall, on)

        values_dict = values_overall.to_dict()

        return values_dict

    def _sanity_check_overall(self, error_types):
        total_errors = error_types.sum()
        n_reports = self.results['report_name_short'].nunique()
        # year-range in ground truth: 2013-2022
        year_range = 2022-2013+1
        # 4 types of scopes: 1, 2mb, 2lb, 3
        n_scopes = 4
        # some reports have more values for more years ==> minimum
        min_total_entries = n_reports * year_range * n_scopes
        if total_errors < min_total_entries:
            raise ValueError(
                "Total number of errors is smaller than total number of entries "
                "in the ground truth data")
        return

    def _compute_metrics_per_doc(self, merged_data, on="value"):
        """Compute metrics per document."""

        all_errors = [f'true_negative_{on}', f'false_positive_{on}',
                      f'true_positive_{on}', f'false_negative_{on}']
        values_per_doc = (
            merged_data
            .groupby('report_name')['error_value']
            .value_counts()
            .unstack(fill_value=0)
            # Ensure all error types are present
            .reindex(columns=all_errors, fill_value=0)
        )

        # TODO: implement
        # self._sanity_check_per_doc(values_per_doc, merged_data)

        values_per_doc['precision_value'] = values_per_doc.apply(
            evaluate_helpers.compute_precision, axis=1, on=on)
        values_per_doc['recall_value'] = values_per_doc.apply(
            evaluate_helpers.compute_recall, axis=1, on=on)
        values_per_doc['f1_value'] = values_per_doc.apply(
            evaluate_helpers.compute_f1, axis=1, on=on)

        return values_per_doc

    def _save_results(self,
                      results,
                      path_to_results: Optional[str],
                      file_name: str):
        """Save the results to a file."""

        if path_to_results is None:
            path_to_results = self.path_to_results

        output_path = os.path.join(self.path_to_results, file_name)
        results.to_csv(output_path)


def evaluate(path_to_results: str, gold_standard: str, mode: str):
    """Sets up path to ground truth and runs evaluation routine.
    
    Args:
        path_to_results: Path to the extraction results directory.
        gold_standard: Path to the gold standard CSV file.
        mode: Evaluation mode ('default', 'precision_recall_f1', or 'both').
    """
    # Verify gold standard file exists
    if not os.path.exists(gold_standard):
        raise FileNotFoundError(
            f"Gold standard file not found at '{gold_standard}'. "
            "Please provide a valid path to the gold standard CSV file."
        )
    
    path_to_ground_truth = gold_standard

    evaluation_metrics = None

    default_evaluator = EvaluatorDefault(
        path_to_results=path_to_results, path_to_ground_truth=path_to_ground_truth)
    default_evaluation_metrics = default_evaluator.run()

    precision_evaluator = EvaluatorPrecisionRecallF1(
        path_to_results=path_to_results,
        path_to_ground_truth=path_to_ground_truth)
    precision_evaluation_metrics = precision_evaluator.run()

    if mode == 'default':
        return default_evaluation_metrics

    elif mode == 'precision_recall_f1':
        return precision_evaluation_metrics

    # Combine default and precision evaluation metrics
    evaluation_metrics = default_evaluation_metrics | precision_evaluation_metrics

    return evaluation_metrics


if __name__ == "__main__":
    evaluate(
        "output/290144e0b01b493390c2c466a87ad1f4",
        "./data/evaluation_dataset/gist_2025.csv",
        "both"
    )
