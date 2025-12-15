"""Parameter dataclasses for the information extraction pipeline."""
from dataclasses import dataclass, field, asdict, fields
import json
import os
from typing import Any, Dict, List


@dataclass
class MlflowParams:
    """Parameters for the MLflow experiment."""
    mlflow_experiment_path: str = field(default=None)
    mlflow_run_name: str = field(default='test_run')

    def construct_mlflow_run_name(self, params_list: List[Dict[str, str]]) -> str:
        """Construct the MLflow run name based on the parameters."""
        config_params, experiment_params = params_list
        in_sample = "sample_139" \
            if config_params.in_sample == "sample_160" \
            else config_params.in_sample
        run_name = (
            f"{experiment_params.llm_params.prompt_type}_"
            f"{experiment_params.pipeline_params.input_mode}_"
            f"{experiment_params.llm_params.llm_model}_"
            f"{experiment_params.semantic_search_params.emb_model}"
        )
        self.mlflow_run_name = run_name

    @staticmethod
    def filter_params(params):
        """Filter out parameters that should not be logged to MLflow."""
        params_dict = asdict(params)
        filtered_params = {
            k: v for k, v in params_dict.items()
            if not next((f.metadata.get('mlflow_log') is False
                         for f in fields(params) if f.name == k), False)
        }
        return filtered_params


@dataclass
class ConfigParams:
    """Parameters for the configuration."""
    evaluation_mode: str = 'default'  # 'default' or 'precision_recall_f1' or 'both'

    gold_standard: str = field(default=None)
    in_sample: str = field(default=None)
    filename_list: list[str] = field(default=None)
    path_to_pdfs: str = field(default=None)

    def update_class(self, updates: Dict[str, Any]):
        """Update class attributes based on a dictionary."""
        for key, value in updates.items():
            if key == "evaluation_mode":
                setattr(self, key, value)
            elif key == "filename_list":
                self.filename_list = updates["filename_list"]
                # If `filename_list` is provided, infer `gold_standard` and `in_sample`
                self._set_gs_and_sample_from_filenames()
            elif "in_sample" in updates:
                # If `in_sample` and optionally `gold_standard` are provided, get `filename_list`
                self.gold_standard = updates.get("gold_standard")
                self.in_sample = updates["in_sample"]
                self.path_to_pdfs = updates.get("path_to_pdfs")
                self.filename_list = self._retrieve_file_paths(
                    self.gold_standard, self.in_sample, self.path_to_pdfs)
            else:
                raise ValueError(
                    "Either 'filename_list' or at least 'in_sample' must be specified."
                )

    def _set_gs_and_sample_from_filenames(self):
        with open('./data/docs/pdf_info.json', 'r', encoding='utf-8') as f:
            pdf_info = json.load(f)

        # Use the first filename in the list as a reference
        # assuming all files have the same gold standard and sample
        first_file = self.filename_list[0]
        first_file_basename = os.path.basename(first_file)
        # Look for key in pdf_info that matches the basename
        matching_key = None
        for key in pdf_info.keys():
            if os.path.basename(key) == first_file_basename:
                matching_key = key
                break
        file_info = pdf_info[matching_key] if matching_key else {}

        # Handle empty or missing 'in_gold_standard'
        in_gs = file_info.get('in_gold_standard', [])
        if in_gs:
            self.gold_standard = in_gs[0]
        else:
            self.gold_standard = None

        # Handle empty or missing 'in_sample'
        in_sample = file_info.get('in_sample', [])
        if len(in_sample) > 1:
            self.in_sample = in_sample[1]
        elif in_sample:
            self.in_sample = in_sample[0]
        else:
            self.in_sample = None

    def _retrieve_file_paths(self, in_gold_standard, in_sample, path_to_pdfs=None):
        """
        Retrieve file paths from JSON based on sample and gold standard criteria.

        Args:
        in_sample (str or list, optional): Value(s) to match in 'in_sample' list.
        in_gold_standard (str or list, optional): Value(s) to match in 'in_gold_standard' list.
        path_to_pdfs (str, optional): If given, returned paths will use this directory 
                                        and the original filename.

        Returns:
        list: Matching file paths.
        """
        with open('./data/docs/pdf_info.json', 'r', encoding='utf-8') as f:
            pdf_info = json.load(f)

        # Convert single values to lists for consistent processing
        in_sample = [in_sample] if isinstance(in_sample, str) else in_sample
        in_gold_standard = [in_gold_standard] if isinstance(
            in_gold_standard, str) else in_gold_standard

        matching_paths = []
        for filepath, info in pdf_info.items():
            sample_match = in_sample and any(
                sample in info['in_sample'] for sample in in_sample)
            gold_match = (not in_gold_standard) or any(
                gold in info['in_gold_standard'] for gold in in_gold_standard)

            if sample_match and gold_match:
                if path_to_pdfs:
                    # Take only filename and append new path
                    filename = os.path.basename(filepath)
                    new_path = os.path.join(path_to_pdfs, filename)
                    matching_paths.append(new_path)
                else:
                    matching_paths.append(filepath)

        return matching_paths


@dataclass
class SemanticSearchParams:
    """Parameters for the semantic search."""
    emb_model: str = field(default="text-embedding-ada-002")
    search_query: str = field(default="""What are the total CO2 emissions in different years?
                            Include Scope 1, Scope 2, and Scope 3 emissions if available.""")
    similarity_top_k: int = field(default=7)
    similarity_min_k: int = field(default=4)
    percentile_threshold: int = field(default=95)
    context_window: int = field(default=0)
    search_method: str = field(default="vector_search")


@dataclass
class LLMParams:
    """Parameters for the LLM."""
    llm_model: str = field(default="gpt-4o-2024-11-20")
    prompt_type: str = field(default=None)
    prompt_role: str = field(default=None)
    prompt_KPI_definitions: str = field(default=None)
    prompt_specifications: str = field(default=None)
    year_min: int = field(default=None)
    year_max: int = field(default=None)

    # If True, the pipeline will request per-token log-probabilities from the LLM
    # and compute a value-level confidence score. Defaults to True to enable
    # value probabilities by default
    return_logprobs: bool = field(default=True)


@dataclass
class PipelineParams:
    """Parameters for the pipeline."""
    input_mode: str = field(default='text+table')
    embed_only: bool = field(default=False)


@dataclass
class ExperimentParams:
    """Parameters for the experiment."""
    pipeline_params: PipelineParams = field(default_factory=PipelineParams)
    semantic_search_params: SemanticSearchParams = field(
        default_factory=SemanticSearchParams)
    llm_params: LLMParams = field(default_factory=LLMParams)


def update_dataclass(instance, updates):
    """ Update fields of a dataclass instance based on a dictionary. """
    for key, value in updates.items():
        if hasattr(instance, key):
            setattr(instance, key, value)
