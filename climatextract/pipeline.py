"""Sets up value retriever pipeline and gets emissions."""
import os
from typing import List
import asyncio
import pandas as pd

from dotenv import load_dotenv
import mlflow
from mlflow.entities import SpanType

import climatextract.semantic_search as semantic_search
import climatextract.helpers as helpers
from climatextract.page_text_and_table_extractor import PageTextAndTableExtractor
from climatextract.resolve_duplicates import handle_duplicates_in_output, select_duplicates_in_output

load_dotenv()  # load environment variables from .env file
os.chdir(helpers.get_project_directory(path_to_file="src"))


class ValueRetrieverPipeline():
    """Run RetrieverPipelines for a single document or separately for each document
        in a list of documents.
        Uses text and tables from documents to retrieve emissions."""

    def __init__(self, experiment_params, embed_model, embeddings_repository,
                 search_query, llm, llm_single_prompt):

        self.embed_model = embed_model
        self.search_query = search_query
        self.embeddings_repository = embeddings_repository
        self.semantic_search_params = experiment_params.semantic_search_params

        self.llm = llm
        self.llm_single_prompt = llm_single_prompt  # prompter class instance
        self.prompt_type = experiment_params.llm_params.prompt_type

        self.input_mode = experiment_params.pipeline_params.input_mode
        self.embed_only = experiment_params.pipeline_params.embed_only

        max_concurrent_pdfs = 20
        self.pipeline_semaphore = asyncio.Semaphore(max_concurrent_pdfs)

        self.page_extractor = PageTextAndTableExtractor()

    @mlflow.trace(span_type=SpanType.CHAIN, attributes={"pipeline": "ValueRetrieverPipeline"})
    async def retrieve_values_for_doc_list(self, filename_list: List[str], path_to_results: str):
        """
        Loop over all pathnames, run 'retrieve_values_per_doc()' for each,
        save intermediate results in a nice format and return final results.
        It is implemented as a basic routine to run bulk API calls in a concurrent way.

        : param filename_list: list of paths
        : param path_to_results: path to save results
        : return: a list of tuples, one element for each pathname
            Each list element has two parts:
            - a dictionary with keys 'doc_overview' and 'co2_emissions' where
                - doc_overview, a pandas DataFrame, that contains for each
                (pathname, page)-combination further information: (content, score, LLM response)
                - co2_emissions, a pandas dataFrame that contains extracted values and units 
                for each(pathname, page, year, scope)-combination.
            - a list of invalid outputs
        """
        final_results = []  # This will collect final outcomes

        first_write = True  # Ensure headers are written only once

        tasks = [self.retrieve_values_per_doc(
            filename) for filename in filename_list]

        for future in asyncio.as_completed(tasks):

            result = await future
            intermediate_results, invalid_outputs = result

            if self.embed_only:
                # Skip processing but ensure coroutine was awaited
                continue

            # Skip saving if both results are empty
            if not intermediate_results and not invalid_outputs:
                print("Skipped empty results for a PDF that could not be processed.")
                continue

            # save results
            final_results.append(result)
            save_results([intermediate_results], path_to_results, first_write,
                         results_type="intermediate_results")
            first_write = False

            print(
                f"Processed and appended results for {result[0]['doc_overview']['report_name'][0]}")

        return final_results

    @mlflow.trace(span_type=SpanType.CHAIN, attributes={"pipeline": "ValueRetrieverPipeline"})
    async def retrieve_values_per_doc(self, filename: str):
        """
            Identifies relevant pages, passes context to LLM and retrieves emissions.
            Depending on input_mode, it also extracts tables from relevant pages and 
            passes them to LLM.
        """
        doc = semantic_search.Pdfdoc(filename=filename,
                                     repository=self.embeddings_repository)

        async with self.pipeline_semaphore:
            pdf_embedded = await doc.load_pdf_and_embed_and_save_to_database(
                self.embed_model, self.embed_only)
            # Return empty results for this PDF if it could not be processed
            if not pdf_embedded:
                print(
                    f"PDF {filename} could not be processed and will be skipped.")
                return [], []

            # Finish processing if only embedding is required
            if self.embed_only:
                return [], []

            relevant_pages = doc.retrieve_relevant_pages(
                search_query=self.search_query,
                params=self.semantic_search_params,
                return_df=False)

            if self.input_mode == "text+table":
                relevant_raw_page_contents = await \
                    self.page_extractor.extract_text_and_tables_from_pages(
                        relevant_pages, filename)
            else:
                relevant_raw_page_contents = await self.extract_text_from_pages(relevant_pages)

            raw_results = await self.get_llm_response_from_document_foreach_page(
                doc_relevant_pages=relevant_raw_page_contents)
            table_results, invalid_output = self.transform_llm_output_and_create_tables(
                doc_relevant_pages=relevant_pages, output=raw_results, filename=filename)

            return table_results, invalid_output

    async def extract_text_from_pages(self, relevant_pages):
        """Extract text from relevant pages."""
        return [page.page_content for page in relevant_pages]

    async def get_llm_response_from_document_foreach_page(self, doc_relevant_pages):
        """Loop over all relevant pages of a document and
        extract emissions from each page, concatenate the results."""
        return await asyncio.gather(*(self._get_emissions_from_raw_text(content)
                                      for content in doc_relevant_pages))

    async def _get_emissions_from_raw_text(self, doc_text):
        """Getter function with semaphore."""

        formatted_prompt = await self.llm_single_prompt.prepare_prompt(doc_text)
        response_dict, _ = await self.llm.bound_run_llm(formatted_prompt)

        return response_dict

    def transform_llm_output_and_create_tables(self, doc_relevant_pages, output, filename: str):
        """Parse LLM output, combine it with document-level info and transform it into tables.

        Parsing is dependent on prompt_type. If prompt_type is 'custom_gaia', the output is parsed
        using a Pydantic parser. If prompt_type is empty string, regex parsing.
        """

        emission_tables = list()

        for page, page_extracted_emissions in zip(doc_relevant_pages, output):

            emission_table, invalid_outputs = self.llm_single_prompt.process_llm_output(
                page_extracted_emissions)
            emission_table['page_label'] = page.page_label

            emission_tables.append(emission_table)

        try:

            co2_emissions = pd.concat(emission_tables)

            co2_emissions['report_name'] = filename

            # Normalize units and add standardized value column
            co2_emissions = helpers.get_unit_normalization_mapping(
                co2_emissions, "unit", pipeline_output_flag=True)
            co2_emissions = helpers.get_value_standardization(
                co2_emissions, "value", "unit_normalized")

            resp_summary = pd.DataFrame({
                'report_name': [filename for _ in doc_relevant_pages],
                'page_label': [(page.page_label)
                               for page in doc_relevant_pages],
                'page_retrieval_scores': [(page.vector_similarity)
                                          for page in doc_relevant_pages],
                'page_text': [(page.page_content) for page in doc_relevant_pages],
                'llm_response': [
                    self.llm_single_prompt.process_llm_output(page_extracted_emissions)[0][
                        "raw_llm_response"]
                    for page_extracted_emissions in output]
            })

            result = {'doc_overview': resp_summary,
                      'co2_emissions': co2_emissions}

            return result, invalid_outputs

        except ValueError as e:

            print(f"Error in {filename}: {e}")

            return None, None


class FileConfig:
    """Configuration to select input files and determine folder for output files."""

    @staticmethod
    def get_path_to_results(run_id: str):
        """Return the path to the data/output/<run_id > folder. 
        Create folder if not existent"""

        path_to_results = os.path.join(
            'output', run_id)
        if not os.path.exists(path_to_results):
            os.makedirs(path_to_results)

        return path_to_results


def save_results(raw_results, path_to_results: str, first_write: bool, results_type: str):
    """Save the results in a nice format."""

    query_responses = concat_prepare_document_wide_information_from(
        raw_results)
    co2_emission_table = concat_prepare_co2emissions_from(raw_results)
    co2_emission_table2_w_query_responses = merge_and_prepare_single_output_table_from(
        query_responses, co2_emission_table)

    if results_type == "intermediate_results":
        output_file = os.path.join(path_to_results, "intermediate_results.csv")

        # Append results iteratively to CSV
        co2_emission_table2_w_query_responses.to_csv(
            output_file, mode='a', header=first_write, index=False)
        return

    # Handle duplicates in final output
    dupl_columns = [
        'report_name_short',
        'extracted_scope_from_llm',
        'extracted_year_from_llm',
        'page_number_used_by_llm',
        'extracted_value_from_llm',
        'normalized_unit_from_dictionary']
    co2_emission_table2_w_query_responses_marked = handle_duplicates_in_output(
        co2_emission_table2_w_query_responses, dupl_columns, select=False)

    # Save final original results
    co2_emission_table2_w_query_responses_marked.to_csv(
        os.path.join(path_to_results,
                     "03_co2_emission_table2_w_query_responses.csv"), index=False)

    # Also save filtered results
    co2_emission_table2_w_query_responses_filtered = select_duplicates_in_output(
        co2_emission_table2_w_query_responses_marked)
    co2_emission_table2_w_query_responses_filtered.to_csv(
        os.path.join(path_to_results,
                     "03_co2_emission_table2_w_query_responses_filtered.csv"), index=False)

    # save results in more compact format (long, including duplicates)
    long_format_df = prepare_long_format_output_table_from(
        co2_emission_table2_w_query_responses_marked)
    long_format_df.to_csv(
        os.path.join(path_to_results, "results_long_format.csv"), index=False)

    # save results in more compact format (wide, deduplicated)
    wide_format_df = prepare_wide_formate_output_table_from(
        co2_emission_table2_w_query_responses_filtered, dupl_columns)
    if wide_format_df is not None:
        wide_format_df.to_csv(
            os.path.join(path_to_results, "results_wide_format.csv"), index=False)

    return co2_emission_table2_w_query_responses


def concat_prepare_document_wide_information_from(raw_results: List[dict]):
    """Concatenate the page metadata and the LLM responses of all reports 
    into a single dataframe and rename columns."""

    try:
        query_responses = pd.concat([report_results['doc_overview']
                                    for report_results in raw_results
                                    if report_results and
                                    report_results.get('doc_overview') is not None],
                                    ignore_index=True)
        query_responses = query_responses.rename(
            columns={"page_label": "page_number_to_llm",
                     "page_retrieval_scores": "page_retrieval_scores",
                     "page_text": "page_texts_to_llm",
                     "llm_response": "text_response_from_llm"})
    except TypeError as e:
        print(f"Error: {e}")
        query_responses = pd.DataFrame()

    return query_responses


def concat_prepare_co2emissions_from(raw_results: List[dict]):
    """Concatenate the emissions data of all reports into a single dataframe and rename columns."""
    co2_emission_table = pd.concat(
        [report_results['co2_emissions'] for report_results in raw_results
         if report_results and 'co2_emissions' in report_results], ignore_index=True)
    co2_emission_table = co2_emission_table.rename(
        columns={"page_label": "page_number_used_by_llm",
                 "year": "extracted_year_from_llm_orig",
                 "extracted_year_from_llm": "extracted_year_from_llm",
                 "scope": "extracted_scope_from_llm_orig",
                 "extracted_scope_from_llm": "extracted_scope_from_llm",
                 "value": "extracted_value_from_llm_orig",
                 "extracted_value_from_llm": "extracted_value_from_llm",
                 "unit": "extracted_unit_from_llm",
                 "unit_normalized": "normalized_unit_from_dictionary"})

    return co2_emission_table


def merge_and_prepare_single_output_table_from(query_responses: pd.DataFrame,
                                               co2_emission_table: pd.DataFrame):
    """Merge the two dataframes and add a column with the short report name."""
    co2_emission_table2_w_query_responses = pd.merge(co2_emission_table, query_responses,
                                                     left_on=[
                                                         "report_name", "page_number_used_by_llm"],
                                                     right_on=["report_name", "page_number_to_llm"])
    co2_emission_table2_w_query_responses["report_name_short"] = [
        os.path.basename(file) for file in co2_emission_table2_w_query_responses.report_name]

    page_numbers_tried = co2_emission_table2_w_query_responses.groupby(
        'report_name')['page_number_to_llm'].unique().apply(list).reset_index()
    page_numbers_tried = page_numbers_tried.rename(
        columns={"page_number_to_llm": "page_numbers_tried_by_llm"})
    page_numbers_tried['automatic_extraction_tried'] = True
    co2_emission_table2_w_query_responses = co2_emission_table2_w_query_responses.merge(
        page_numbers_tried, on='report_name', how='left')

    return co2_emission_table2_w_query_responses


def prepare_long_format_output_table_from(
        co2_emission_table2_w_query_responses_marked: pd.DataFrame) -> pd.DataFrame:
    """Prepare a long format output table."""
    # Make a copy to avoid mutating original data
    df = co2_emission_table2_w_query_responses_marked.copy()

    # Remove rows where extracted_value_from_llm is missing
    df = df.dropna(subset=['extracted_value_from_llm'])
    # Rename columns to final output names
    renames = {
        'report_name_short': 'report_id',
        'extracted_year_from_llm': 'year',
        'standardized_value': 'value_std',
        'extracted_value_from_llm': 'value_raw',
        'value_probability': 'value_score',
        'extracted_unit_from_llm': 'unit_raw',
        'unit_probability': 'unit_score',
        'normalized_unit_from_dictionary': 'unit_cat',
        'duplicate_flag': 'dupl_flag',
        'select_flag': 'select_flag',
        'page_number_used_by_llm': 'page'
    }

    df = df.rename(columns=renames)

    # Create indicator field by joining "scope" and the scope column
    df['indicator'] = "scope " + df['extracted_scope_from_llm'].astype(str)

    # Assign 't CO2e' to unit_std if "scope" is in indicator, else None
    df['unit_std'] = df['indicator'].apply(
        lambda x: 't CO2e' if 'scope' in str(x) else None)

    # Final output columns in requested order
    target_cols = [
        'report_id', 'year', 'indicator', 'value_std', 'value_raw', 'value_score',
        'unit_std', 'unit_raw', 'unit_score', 'unit_cat', 'dupl_flag', 'select_flag', 'page'
    ]
    long_format_df = df[target_cols]
    return long_format_df


def prepare_wide_formate_output_table_from(
        co2_emission_table2_w_query_responses_filtered: pd.DataFrame,
        col_names: List[str]) -> pd.DataFrame:
    """Prepare a wide format output table."""
    # Make a copy to avoid mutating original data
    df = co2_emission_table2_w_query_responses_filtered.copy()
    # Remove rows with no meaningful extracted value
    df = df[df['extracted_value_from_llm'].notnull() & (
        df['extracted_value_from_llm'] != "Nothing extracted. No Regex match")]

    # Check if there are still duplicates
    dupl_columns = col_names[:3]
    duplicates = df.duplicated(subset=dupl_columns, keep=False)
    if duplicates.any():
        print(
            "Warning: There are still duplicates in the filtered output, \
                no wide format output produced!"
        )
        print("Duplicate rows:")
        print(df[duplicates][dupl_columns])
        return None

    # Pivot the DataFrame to wide format

    renames = {
        'report_name_short': 'report_id',
        'extracted_year_from_llm': 'year',
        'standardized_value': 'value_std',
        'extracted_value_from_llm': 'value_raw',
        'value_probability': 'value_score',
        'extracted_unit_from_llm': 'unit_raw',
        'unit_probability': 'unit_score',
        'normalized_unit_from_dictionary': 'unit_cat',
        'page_number_used_by_llm': 'page'
    }
    df = df.rename(columns=renames)

    # Adjust dupl_reason based on value_raw presence
    def _adjust_dupl_reason(row):
        if pd.isna(row['value_raw']) or row['value_raw'] == "Nothing extracted. No Regex match":
            return None
        if pd.isna(row.get('dupl_reason', None)):
            return 0
        return row['dupl_reason']

    df['dupl_reason'] = df.apply(_adjust_dupl_reason, axis=1)

    # Pivot table: one row per report_id and year, columns per scope type
    pivot_cols = ['value_std', 'value_raw', 'value_score',
                  'unit_raw', 'unit_score', 'unit_cat', 'page', 'dupl_reason']
    df_pivot = df.pivot_table(
        index=['report_id', 'year'],
        columns='extracted_scope_from_llm',
        values=pivot_cols,
        aggfunc='first'  # or appropriate aggregation if multiples exist
    )

    # Flatten MultiIndex columns to single level with naming scheme
    df_pivot.columns = [
        f"scope_{scope}_{col}" for col, scope in df_pivot.columns]
    df_pivot = df_pivot.reset_index()

    # Add unit_std columns per scope if value_std present
    scopes = sorted(df['extracted_scope_from_llm'].dropna().unique())
    for s in scopes:
        val_col = f"scope_{s}_value_std"
        unit_std_col = f"scope_{s}_unit_std"
        df_pivot[unit_std_col] = df_pivot[val_col].apply(
            lambda x: "t CO2e" if pd.notna(x) else None)

    # Reorder columns as requested
    def _reorder_columns(df_local):
        base_cols = ['report_id', 'year']
        suffixes = ['value_std', 'value_raw', 'value_score', 'unit_std',
                    'unit_raw', 'unit_score', 'unit_cat', 'dupl_reason', 'page']
        ordered_cols = base_cols + \
            [f'scope_{scope}_{suffix}' for suffix in suffixes for scope in scopes]
        existing_cols = [
            col for col in ordered_cols if col in df_local.columns]
        return df_local[existing_cols]

    long_format_df = _reorder_columns(df_pivot)

    return long_format_df
