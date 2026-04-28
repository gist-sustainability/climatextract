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
    # No default — embedding handler falls back to its class ``MODEL``
    # attribute (provider-specific) when this isn't set.
    emb_model: str = field(default=None)
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

    # Maximum number of concurrent embedding API calls.
    max_parallel_embedding_calls: int = field(default=5)


@dataclass
class LLMParams:
    """Parameters for the LLM. TOML overrides these defaults."""
    # No default — LLM handler falls back to its class ``MODEL``
    # attribute (provider-specific) when this isn't set.
    llm_model: str = field(default=None)
    prompt_type: str = field(default=None)
    prompt_role: str = field(default=None)
    prompt_KPI_definitions: str = field(default=None)
    prompt_specifications: str = field(default=None)
    year_min: int = field(default=None)
    year_max: int = field(default=None)

    # If True, the pipeline requests per-token log-probabilities from
    # the LLM and computes a value-level confidence score.
    return_logprobs: bool = field(default=True)

    # Sampling temperature.
    temperature: float = field(default=0.0)

    # Reasoning effort for reasoning models (o1/o3/gpt-5 family).
    # ``"none"`` suppresses reasoning. LiteLLM's ``drop_params`` removes
    # this param when calling models that don't support it.
    reasoning_effort: str = field(default="none")

    # Maximum number of concurrent LLM API calls.
    max_parallel_llm_prompts_running: int = field(default=4)


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
