"""Parameter dataclasses for the information extraction pipeline."""
from dataclasses import dataclass, field, asdict, fields


@dataclass
class MlflowParams:
    """Parameters for the MLflow experiment."""
    mlflow_experiment_path: str = field(default=None)
    mlflow_run_name: str = field(default='test_run')

    def construct_mlflow_run_name(self, params_list) -> str:
        """Construct the MLflow run name based on the parameters."""
        config_params, experiment_params = params_list
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
    gold_standard: str = field(default=None)
    filename_list: list[str] = field(default=None)


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
    # Path to custom embeddings repository. If None, uses default path:
    # data/processed/embeddings/{emb_model}_from_2025_03_06.duckdb
    embeddings_repository: str = field(default=None)

    # Maximum number of concurrent embedding API calls. If None, the
    # adapter falls back to its per-model default (see
    # climatextract.adapters.azure_openai._DEFAULT_MAX_CONCURRENT).
    max_parallel_embedding_calls: int = field(default=None)


@dataclass
class LLMParams:
    """Parameters for the LLM."""
    # TODO: no default llm model
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

    # Maximum number of concurrent LLM API calls. If None, uses model-specific defaults:
    max_parallel_llm_prompts_running: int = field(default=None)


@dataclass
class PipelineParams:
    """Parameters for the pipeline."""
    input_mode: str = field(default='text')
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
