"""Defines prompts with respective parsing methods"""
# very unclear if this file name & content is a good structure
from typing import List, Optional, Literal, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import re
import json
import pandas as pd

from pydantic import BaseModel, Field, confloat, ValidationError
from llama_index.core.output_parsers import PydanticOutputParser
import mlflow
from mlflow.entities import SpanType
from llama_index.core import PromptTemplate


import climatextract.helpers as helpers
import math, bisect
from itertools import accumulate

class PromptProcessorInterface(ABC):
    """Abstract base class defining the interface for all prompt processors.
    
    This serves as a strict contract that all prompt processor classes must implement.
    Python will raise TypeError if a subclass doesn't implement all abstract methods.
    """
    
    @abstractmethod
    async def prepare_prompt(self, doc_text: str) -> str:
        """Prepare the formatted prompt for the LLM.
        
        Args:
            doc_text: The document text to process
            
        Returns:
            Formatted prompt string ready for the LLM
        """
        pass
    
    @abstractmethod
    def process_llm_output(self, llm_output) -> Tuple[pd.DataFrame, List]:
        """Process LLM output and return structured data.
        
        Args:
            llm_output: Raw output from the LLM
            
        Returns:
            Tuple of (processed_dataframe, invalid_outputs)
        """
        pass
    
    @abstractmethod
    def _reformat_output_table(self, output_list: List, llm_output: str, log_blocks=None) -> pd.DataFrame:
        """Change output table to desired format.
        
        Args:
            output_list: List of processed entries
            llm_output: Raw LLM output string
            
        Returns:
            Formatted DataFrame with standardized columns
        """
        pass
    
    @abstractmethod
    def _fill_no_extractions_table(self) -> pd.DataFrame:
        """Create table when nothing was extracted.
        
        Returns:
            DataFrame with default structure when no data is found
        """
        pass

class KpiEntry(BaseModel):
    """Data model for a KPI entry extracted from the LLM output."""
    year: Optional[int] = Field(
        None,
        description="Year of KPI extracted, e.g. 2021, 2020, 2019.",
        example=2020
    )

    kpi_name: Optional[Literal["1", "2 (market-based)", "2 (location-based)", "3"]] = Field(
        None,
        description="Scope category of the GHG emission. Only possible values are: 1 for Scope 1, 2 (market-based) and 2 (location-based) for Scope 2, and 3 for Scope 3.",
        example=1
    )

    value: Optional[confloat(ge=0)] = Field(
        None,
        description="The numerical value of the KPI, e.g., 35.7, 44000 or 'null' if not available",
        example=44000.0
    )

    unit: Optional[str] = Field(
        None,
        description="Unit of the value e.g., 'tCO2e', 'kg', 't', 'tonnes', or 'null' if not available.",
        example="tCO2e"
    )

class KpiEntries(BaseModel):
    """Data model for a list of KPI entries extracted from the LLM output."""
    KPI_Entries: List[KpiEntry] = Field(description="List of KPI entries")

class PromptRoleAndTask:
    """Describes LLM role and task for prompt."""

    # role: str = helpers.read_txt_file('./prompt/role_task/role_task_1.txt')
    role: str = (
        "You are a climate analyst tasked with extracting specific absolute numerical data from corporate reports. \n"
        "Your objective is to extract only the absolute values for the following Key Performance Indicators (KPIs) related to CO2 emissions across the entire company.\n\n"
    )


class PromptKpiDefinitions:
    """Provides definitions to each KPI in prompt."""

    # definitions_string: str = helpers.read_json_to_str(
    #     './prompt/kpi_definitions/scope_12mb2lb3_short.json')
    definitions_string: str = (
        "Scope 1: Scope 1 CO2 Emissions: Direct GHG emissions from sources owned or controlled by the organization (e.g., fuel combustion, company-owned vehicles). \n"
        "Scope 2 (market-based): Scope 2 (market-based) CO2 Emissions: Indirect GHG emissions from purchased energy, calculated based on energy procurement choices (e.g., renewable energy contracts). \n"
        "Scope 2 (location-based): Scope 2 (location-based) CO2 Emissions: Indirect GHG emissions from purchased energy, calculated using the average emissions intensity of the local electricity grid. \n"
        "Scope 3: Scope 3 CO2 Emissions: Indirect GHG emissions from the organization's value chain, both upstream and downstream (e.g., supply chain, business travel, product use). \n"
    )


class PromptSpecifications:
    """Describes the specifications for the prompt."""

    # specifications: str = helpers.read_txt_file(
    #     './prompt/specifications/specifications_1.txt')
    specifications: str = (
        "Only extract values which refer to the whole company.\n"
        "Only extract absolute values representing total CO2 emissions (e.g., in tons).\n"
        "Do not extract any relative values such as percentages, year-over-year changes, or trends. Ignore all values that are expressed as percentages (%) or involve relative comparisons (e.g., increases, decreases, or changes over time).\n"
        "Footnotes or annotations in metric names should be treated as references and ignored for the extraction process. \n"
        "Do not modify values based on footnotes, annotations, or any external data sources.\n"
        "Do not perform any calculations or transformations on the values. Extract and report the data exactly as presented. Do not invent values.\n"
        "Ensure your extraction only includes absolute values for the defined KPIs, strictly following these guidelines. \n"
        "If the subtype \"market-based\" or \"location-based\" is not mentioned for a value of Scope 2, always assume that the value refers to the KPI Scope 2 (location-based).\n"
        "Do not extract all subcategories of Scope 3, but only total values if total values are available for Scope 3. \n"
        "Only extract value referring to Scope 1, Scope 2 or Scope 3 separately. Do not extract values representing a sum of any the scopes.\n"
    )


class CustomPromptGaia(PromptProcessorInterface):
    """Implements PromptProcessorInterface using Pydantic parsing.
    Strategy: We make a single query to the LLM to extract all the Scope 1, 2, 3 values."""

    role: Optional[str] = field(default=PromptRoleAndTask.role)
    kpi_definitions: Optional[str] = field(
        default=PromptKpiDefinitions().definitions_string)
    specifications: Optional[str] = field(
        default=PromptSpecifications.specifications)

    def __init__(self, prompt_params):
        """Optional prompt parameters allow default values to be loaded from a file if None is provided."""
        if prompt_params.prompt_role is None:
            role = PromptRoleAndTask.role  # Default role definition
        else:
            role = prompt_params.prompt_role

        if prompt_params.prompt_KPI_definitions is None:
            kpi_definitions = PromptKpiDefinitions().definitions_string
        else:
            kpi_definitions = prompt_params.prompt_KPI_definitions

        if prompt_params.prompt_specifications is None:
            specifications = PromptSpecifications.specifications  # Default specifications
        else:
            specifications = prompt_params.prompt_specifications

        if prompt_params.year_min is None:
            min_year = 2010
        else:
            min_year = prompt_params.year_min

        if prompt_params.year_max is None:
            max_year = 2024
        else:
            max_year = prompt_params.year_max

        self.query = f'{role}\n{kpi_definitions}\n{specifications} \n \
        Year range for the search: only extract values from {min_year} to {max_year}.\n\n \
        Here is the excerpt: \n {{context_str}}'

        self.parser = PydanticOutputParser(output_cls=KpiEntries)

    async def prepare_prompt(self, doc_text: str) -> str:
        prompt = self.query
        parsing_instruction = self.parser

        prompt_tmpl = PromptTemplate(template=f"{prompt}",
                                         output_parser=parsing_instruction)
        formatted_prompt = prompt_tmpl.format(context_str=doc_text)

        return formatted_prompt

    @mlflow.trace(span_type=SpanType.CHAIN, attributes={"query": "parser"})
    def process_llm_output(self, llm_output) -> Tuple[pd.DataFrame, List]:
        """Parse JSON output and compute confidence if log-probs are available."""

        invalid_kpi_entries: List = []

        # Split dict vs legacy string
        if isinstance(llm_output, dict):
            raw_output_str = str(llm_output.get("content", ""))
            logprobs_object = llm_output.get("logprobs")
            log_blocks = logprobs_object.content if logprobs_object else None
        else:
            raw_output_str, log_blocks = str(llm_output), None

        try:
            content = raw_output_str.strip("```json").strip("```")
            content = json.loads(content)
            valid_kpi_entries, invalid_kpi_entries = self._validate_llm_output_content(
                content)
            output_table = self._reformat_output_table(
                valid_kpi_entries, raw_output_str, log_blocks)

        except Exception as e:
            # RateLimitError, APIStatusError, RuntimeError
            print(f"Error while processing LLM output: {e}")
            emtpy_table = self._fill_no_extractions_table()
            output_table = self._reformat_output_table(emtpy_table, raw_output_str, log_blocks)

        return output_table, invalid_kpi_entries

    def _validate_llm_output_content(self, llm_output):
        """Process the output of the LLM to validate the KPI entries. """

        valid_entries = []
        invalid_entries = []

        for entry in llm_output["KPI_Entries"]:
            try:
                validated_entry = KpiEntry(**entry)
                entry_dict = vars(validated_entry)
                # Store original value from the raw JSON entry to use for confidence calculation,
                # as the Pydantic model might coerce the type.
                entry_dict['value_original'] = entry.get('value')
                entry_dict['unit_original'] = entry.get('unit')
                valid_entries.append(entry_dict)
            except ValidationError as e:
                invalid_entries.append({"entry": entry, "error": str(e)})

        return valid_entries, invalid_entries

    def _reformat_output_table(self, output_list: List, llm_output: str, log_blocks=None) -> pd.DataFrame:
        """Change output table to desired format"""

        if len(output_list) > 0:
            output_table = pd.DataFrame(output_list)
            output_table = output_table.rename(columns={"kpi_name": "scope"})
        else:
            output_table = self._fill_no_extractions_table()

        def convert_values(x):
            if x == "1":
                return "1"
            elif x == "2 (market-based)":
                return "2mb"
            elif x == "2 (location-based)":
                return "2lb"
            elif x == "3":
                return "3"
            else:
                return pd.NA

        output_table["extracted_scope_from_llm"] = output_table.scope.apply(
            convert_values)
        output_table["extracted_year_from_llm"] = pd.to_numeric(
            output_table.year, errors='coerce')
        output_table["extracted_value_from_llm"] = pd.to_numeric(
            output_table.value, errors='coerce')

        # Compute probability when possible, using the original un-cleaned value string
        if log_blocks is not None and len(output_list) > 0:
            output_table["value_probability"] = output_table.apply(
                lambda r: helpers.compute_value_confidence(
                    r.get('value_original', r['value']),  # Use the original value for searching
                    log_blocks
                ),
                axis=1
            )

            output_table["unit_probability"] = output_table.apply(
                lambda r: helpers.compute_value_confidence(
                    r.get('unit_original', r['unit']),  # Use the original unit for searching
                    log_blocks
                ),
                axis=1
            )

            # Drop the temporary columns
            for col in ("value_original", "unit_original"):
                if col in output_table.columns:
                    output_table = output_table.drop(columns=[col])

        output_table["raw_llm_response"] = llm_output

        return output_table

    def _fill_no_extractions_table(self) -> pd.DataFrame:
        """Create table when nothing was extracted."""

        output_table_dict = {'year': [str(num) for num in range(2010, 2026)],
                             'scope': ["1", "2 (market-based)", "2 (location-based)", "3"]}
        output_table = helpers.expand_grid(output_table_dict)
        output_table['value'] = "Nothing extracted. No Regex match"
        output_table['unit'] = "Nothing extracted. No Regex match"
        return output_table


class LlmSinglePromptQueryScope123(PromptProcessorInterface):
    """Implements PromptProcessorInterface using regex parsing.
    Strategy: We make a single query to the LLM to extract all the Scope 1, 2, 3 values.
    """

    def __init__(self, prompt_params):
        self.min_year = prompt_params.year_min if prompt_params.year_min is not None else 2010
        self.max_year = prompt_params.year_max if prompt_params.year_max is not None else 2024
        self.query = None
    
    async def prepare_prompt(self, doc_text: str) -> str:
        parts = [
            "Extract key pieces of information from this sustainability report.",
            "If a particular piece of information is not present, output 'Not specified'.",
            "If the report does not mention if Scope 2 is location-based or market-based, assume it is Scope 2 location-based. Do not extract it as Scope 2 market-based.",
            "",
            "Use the following format:",
            "0. What is the title"
        ]

        index = 1
        for year in range(self.min_year, self.max_year + 1):
            parts.append(f"{index}. What are the Scope 1 emissions in {year}")
            index += 1

        for year in range(self.min_year, self.max_year + 1):
            parts.append(f"{index}. What are the Scope 2 emissions in {year}")
            index += 1

        for year in range(self.min_year, self.max_year + 1):
            parts.append(f"{index}. What are the Scope 3 emissions in {year}")
            index += 1

        parts.append("")
        parts.append("Document:\n{context_str}")

        # example response at the end of the prompt
        parts.append("")
        parts.append("0. What is the title: Our responsibility. Report {self.min_year}")
        parts.append(f"1. What are the Scope 1 emissions in {self.min_year}: <value> <unit>")
        parts.append(f"2. What are the Scope 1 emissions in {self.min_year + 1}: <value> <unit>")

        self.query = "\n".join(parts)

        prompt_tmpl = PromptTemplate(self.query)
        formatted_prompt = prompt_tmpl.format(context_str=doc_text)

        return formatted_prompt

    def process_llm_output(self, llm_output) -> Tuple[pd.DataFrame, List]:
        """Extract year, scope, value, unit and compute probability if available."""

        # Accept both legacy string and new dict with logprobs
        if isinstance(llm_output, dict):
            raw_str = str(llm_output.get("content", ""))
            logprobs_object = llm_output.get("logprobs")
            log_blocks = logprobs_object.content if logprobs_object else None
        else:
            raw_str, log_blocks = str(llm_output), None

        output_list: List = []
        invalid_output: List = []

        output_list, invalid_output = self._find_match(
            raw_str, output_list, invalid_output, information_found=True, log_blocks=log_blocks)
        output_list, invalid_output = self._find_match(
            raw_str, output_list, invalid_output, information_found=False, log_blocks=log_blocks)

        output_table = self._reformat_output_table(output_list, raw_str)
        return output_table, invalid_output

    def _find_match(self, llm_output: str, output_list: List, invalid_output: List, 
                    information_found: bool, log_blocks=None) -> Tuple[List, List]:
        """Use regex pattern matching to extract information for specific year
        and scope if available"""

        # In case we did find information for a specific year and scope, GPT should provide it in the following format
        # pattern = r'What are the Scope ([123]{1}) emissions in (20[12]\d): ([0-9\.,]+) (.{0,50})'
        # the previous expression does not work if no unit is provided
        if information_found:
            pattern = r'What are the Scope ([123]{1}) emissions in (20[12]\d): ([0-9\.,]+)( (.{0,50})|\\n|\n)'
        # In case we did NOT find information for a specific year and scope,
        # our LLM query asks GPT to provide it in the following format
        else:
            pattern = r'What are the Scope ([123]{1}) emissions in (20[12]\d): (Not specified)$'

        matches = re.finditer(pattern, llm_output, re.MULTILINE)

        if matches is None:
            invalid_output.append("Regex failed")

        for _, match in enumerate(matches, start=1):
            if information_found:
                original_value = match.group(3)
                cleaned_value = helpers.remove_decimal_commas_in_numbers(original_value)

                # Compute value and unit probabilities using helper function
                probability = helpers.compute_substring_probability(match, 3, log_blocks)
                unit_probability = helpers.compute_substring_probability(match, 5, log_blocks)

                entry = pd.DataFrame.from_dict({
                    "year": [match.group(2)],
                    "scope": [match.group(1)],
                    "value": [cleaned_value],
                    "unit": [match.group(5)],
                    "value_probability": [probability],
                    "unit_probability": [unit_probability],
                })
            else:
                entry = pd.DataFrame.from_dict({
                    "year": [match.group(2)],
                    "scope": [match.group(1)],
                    "value": [match.group(3)],
                    "unit": [match.group(3)]
                })
            output_list.append(entry)

        return output_list, invalid_output

    def _reformat_output_table(self, output_list: List, llm_output: str) -> pd.DataFrame:
        """Change output table to desired format"""

        if len(output_list) > 0:
            output_table = pd.concat(output_list, ignore_index=True)
        else:
            output_table = self._fill_no_extractions_table()

        output_table["extracted_year_from_llm"] = pd.to_numeric(
            output_table.year, errors='coerce')
        output_table["extracted_value_from_llm"] = pd.to_numeric(
            output_table.value, errors='coerce')

        output_table["raw_llm_response"] = llm_output

        return output_table

    def _fill_no_extractions_table(self) -> pd.DataFrame:
        """Create table when nothing was extracted."""

        output_table_dict = {'year': [str(num) for num in range(2010, 2026)],
                             'scope': [str(num) for num in range(1, 4)]}
        output_table = helpers.expand_grid(output_table_dict)
        output_table['value'] = "Nothing extracted. No Regex match"
        output_table['unit'] = "Nothing extracted. No Regex match"
        return output_table


class LlmSinglePromptQueryScope12lb2mb3(PromptProcessorInterface):
    """Implements PromptProcessorInterface using regex parsing.
    Strategy: We make a single query to the LLM to extract all the Scope 1, Scope 2 (location-based),
    Scope 2 (market-based), Scope 3 values.
    """

    def __init__(self, prompt_params):
        self.min_year = prompt_params.year_min if prompt_params.year_min is not None else 2010
        self.max_year = prompt_params.year_max if prompt_params.year_max is not None else 2024
        self.query = None

    async def prepare_prompt(self, doc_text: str) -> str:
        parts = [
            "Extract key pieces of information from this sustainability report.",
            "If a particular piece of information is not present, output 'Not specified'.",
            "If the report does not mention if Scope 2 is location-based or market-based, assume it is Scope 2 location-based. Do not extract it as Scope 2 market-based.",
            "",
            "Use the following format:",
            "0. What is the title"
        ]

        index = 1
        for year in range(self.min_year, self.max_year + 1):
            parts.append(f"{index}. What are the Scope 1 emissions in {year}")
            index += 1

        for year in range(self.min_year, self.max_year + 1):
            parts.append(f"{index}. What are the Scope 2 (market-based) emissions in {year}")
            index += 1

        for year in range(self.min_year, self.max_year + 1):
            parts.append(f"{index}. What are the Scope 2 (location-based) emissions in {year}")
            index += 1

        for year in range(self.min_year, self.max_year + 1):
            parts.append(f"{index}. What are the Scope 3 emissions in {year}")
            index += 1

        parts.append("")
        parts.append("Document:\n{context_str}")

        # example response at the end of the prompt
        parts.append("")
        parts.append("0. What is the title: Our responsibility. Report {self.min_year}")
        parts.append(f"1. What are the Scope 1 emissions in {self.min_year}: <value> <unit>")
        parts.append(f"2. What are the Scope 1 emissions in {self.min_year + 1}: <value> <unit>")

        self.query = "\n".join(parts)

        prompt_tmpl = PromptTemplate(self.query)
        formatted_prompt = prompt_tmpl.format(context_str=doc_text)

        return formatted_prompt

    def process_llm_output(self, llm_output) -> Tuple[pd.DataFrame, List]:
        """Extract data and compute probability when log-probs available."""

        if isinstance(llm_output, dict):
            raw_str = str(llm_output.get("content", ""))
            logprobs_object = llm_output.get("logprobs")
            log_blocks = logprobs_object.content if logprobs_object else None
        else:
            raw_str, log_blocks = str(llm_output), None

        output_list: List = []
        invalid_output: List = []

        output_list, invalid_output = self._find_match(
            raw_str, output_list, invalid_output, information_found=True, log_blocks=log_blocks)
        output_list, invalid_output = self._find_match(
            raw_str, output_list, invalid_output, information_found=False)

        output_table = self._reformat_output_table(output_list, raw_str)

        return output_table, invalid_output

    def _find_match(self, llm_output: str, output_list: List, invalid_output: List, 
                    information_found: bool, log_blocks=None) -> Tuple[List, List]:
        """Use regex pattern matching to extract information for specific year
        and scope if available"""
        str_llm_output = str(llm_output)

        # In case we did find information for a specific year and scope, GPT should provide it in the following format
        # pattern = r'What are the Scope ([123]{1}) emissions in (20[12]\d): ([0-9\.,]+) (.{0,50})'
        # the previous expression does not work if no unit is provided
        if information_found:
            pattern = r'What are the Scope \b([13]{1}|(2 \(market-based\))|(2 \(location-based\))) emissions in (20[12]\d): ([0-9\.,]+)( (.{0,50})|\\n|\n)'
        # In case we did NOT find information for a specific year and scope,
        # our LLM query asks GPT to provide it in the following format
        else:
            pattern = r'What are the Scope \b([13]{1}|(2 \(market-based\))|(2 \(location-based\))) emissions in (20[12]\d): (Not specified)$'

        matches = re.finditer(pattern, str_llm_output, re.MULTILINE)

        if matches is None:
            invalid_output.append("Regex failed")

        for _, match in enumerate(matches, start=1):
            if information_found:
                original_value = match.group(5)
                cleaned_value = helpers.remove_decimal_commas_in_numbers(original_value)

                # Compute value and unit probabilities using helper function
                probability = helpers.compute_substring_probability(match, 5, log_blocks)
                unit_probability = helpers.compute_substring_probability(match, 7, log_blocks)

                entry = pd.DataFrame.from_dict({
                    "year": [match.group(4)],
                    "scope": [match.group(1)],
                    "value": [cleaned_value],
                    "unit": [match.group(7)],
                    "value_probability": [probability],
                    "unit_probability": [unit_probability],
                })
                
            else:
                entry = pd.DataFrame.from_dict({
                    "year": [match.group(4)],
                    "scope": [match.group(1)],
                    "value": [match.group(5)],
                    "unit": [match.group(5)]
                })
            output_list.append(entry)

        return output_list, invalid_output

    def _reformat_output_table(self, output_list: List, llm_output: str) -> pd.DataFrame:
        """Change output table to desired format"""
        if len(output_list) > 0:
            output_table = pd.concat(output_list, ignore_index=True)
        else:
            output_table = self._fill_no_extractions_table()

        def convert_values(x):
            if x == "1":
                return "1"
            elif x == "2 (market-based)":
                return "2mb"
            elif x == "2 (location-based)":
                return "2lb"
            elif x == "3":
                return "3"
            else:
                return pd.NA

        output_table["extracted_scope_from_llm"] = output_table.scope.apply(
            convert_values)
        output_table["extracted_year_from_llm"] = pd.to_numeric(
            output_table.year, errors='coerce')
        output_table["extracted_value_from_llm"] = pd.to_numeric(
            output_table.value, errors='coerce')

        output_table["raw_llm_response"] = llm_output

        return output_table

    def _fill_no_extractions_table(self) -> pd.DataFrame:
        """Create table when nothing was extracted."""
        output_table_dict = {'year': [str(num) for num in range(2013, 2023)],
                             'scope': ["1", "2 (market-based)", "2 (location-based)", "3"]}
        output_table = helpers.expand_grid(output_table_dict)
        output_table['value'] = "Nothing extracted. No Regex match"
        output_table['unit'] = "Nothing extracted. No Regex match"

        return output_table
