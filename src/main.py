"""Main script to run the pipeline."""
import asyncio
from dataclasses import asdict
import logging
import os
import sys
from typing import Dict, List, Optional

import mlflow

from pipeline import FileConfig, ValueRetrieverPipeline, save_results
from experiment_setup import Experiment
import config
from params import ConfigParams, ExperimentParams, MlflowParams, update_dataclass
from evaluator import evaluate
import semantic_search
import prompts_with_prompt_parsers
from data_lake_manager import DataLakeManager

def main(mlflow_params: Dict[str, str],
         config_params: Dict[str, str],
         experiment_params: Dict[str, str]):
    """Main function to run the pipeline."""

    experiment = Experiment(mlflow_params=mlflow_params)
    mlflow_params.construct_mlflow_run_name([config_params, experiment_params])
    experiment.setup_experiment()
    mlflow.openai.autolog()

    # Initiate the MLflow run context
    with mlflow.start_run(run_name=mlflow_params.mlflow_run_name) as run:
        run_id = run.info.run_id
        path_to_results = FileConfig.get_path_to_results(run_id=run_id)

        mlflow.log_params(MlflowParams.filter_params(mlflow_params))
        mlflow.log_param("mlflow_run_name", mlflow_params.mlflow_run_name)
        mlflow.log_params(asdict(config_params))
        mlflow.log_param("filename_list", config_params.filename_list)
        mlflow.log_params(asdict(experiment_params.pipeline_params))
        mlflow.log_params(
            asdict(experiment_params.semantic_search_params))
        mlflow.log_params(asdict(experiment_params.llm_params))

        # call the extraction pipeline
        extraction_output = extract(
            config_params=config_params,
            experiment_params=experiment_params,
            path_to_results=path_to_results
        )
        if extraction_output is None:
            return run_id

        mlflow.log_param("prompt", extraction_output["prompt"])
        mlflow.log_metrics(extraction_output["llm_costs"])
        # log the results to mlflow

        # call the evaluation pipeline
        if extraction_output["has_results"] and config_params.evaluation_mode != "no_evaluation":
            evaluation_metrics = evaluate(
                path_to_results=path_to_results,
                gold_standard=config_params.gold_standard,
                mode=config_params.evaluation_mode
            )
            if evaluation_metrics is not None:
                mlflow.log_metrics(evaluation_metrics)

        mlflow.log_artifacts(path_to_results)
    return run_id


def extract(
    config_params: ConfigParams,
    experiment_params: ExperimentParams,
    path_to_results: str
) -> Optional[Dict[str, object]]:
    """Run the extraction pipeline and return logging payload."""
    embeddings_repo = semantic_search.EmbeddingsRepository(
        database_name=(
            f"data/processed/embeddings/"
            f"{experiment_params.semantic_search_params.emb_model}"
            f"_from_2025_03_06.duckdb")
    )

    embed_model = config.EmbeddingModel(
        model_name=experiment_params.semantic_search_params.emb_model)
    llm = config.Llm(
        model_name=experiment_params.llm_params.llm_model,
        return_logprobs=experiment_params.llm_params.return_logprobs
    )

    search_query = semantic_search.SearchQuery(
        search_query=experiment_params.semantic_search_params.search_query,
        repository=embeddings_repo)

    storage_account_url = os.environ.get("AZURE_STORAGE_ACCOUNT_URL")
    data_lake_manager = DataLakeManager(storage_account_url)

    if not data_lake_manager.execute_complete_workflow(
        filename_list=config_params.filename_list,
        embeddings_repo=embeddings_repo,
        input_mode=experiment_params.pipeline_params.input_mode
    ):
        print("Data lake workflow failed.")
        return None

    if not search_query.embed_search_query_and_save_to_database(embed_model):
        raise ValueError(
            "Failed to embed the search query and save it to the database.")

    if config_params.gold_standard in (None, "gist_2025"):
        if experiment_params.llm_params.prompt_type == 'custom_gaia':
            llm_single_prompt = prompts_with_prompt_parsers.CustomPromptGaia(
                prompt_params=experiment_params.llm_params)
        else:
            llm_single_prompt = prompts_with_prompt_parsers.LlmSinglePromptQueryScope12lb2mb3(
                prompt_params=experiment_params.llm_params)
    else:
        raise ValueError(
            (
                f"Unsupported gold_standard "
                f"'{config_params.gold_standard}'."
                f"Only 'gist_2025' is supported."
            ))

    retriever_pipeline = ValueRetrieverPipeline(
        experiment_params=experiment_params,
        embed_model=embed_model,
        embeddings_repository=embeddings_repo,
        search_query=search_query,
        llm=llm,
        llm_single_prompt=llm_single_prompt
    )

    results = asyncio.run(
        retriever_pipeline.retrieve_values_for_doc_list(
            filename_list=config_params.filename_list,
            path_to_results=path_to_results
        )
    )
    if not results:
        print("No pdfs were processed. Exiting.")
        return None

    raw_results, invalid_llm_outputs = rearrange_results(results)

    # Create combined token counts from both LLM and embedding model
    llm_costs = llm.create_llm_costs_dict()
    
    # Add embedding tokens from embedding model to the LLM costs
    if hasattr(embed_model, 'token_counter') and hasattr(
            embed_model.token_counter, 'total_embedding_token_count'):
        llm_costs["embedding_tokens"] += embed_model.token_counter.total_embedding_token_count

    # Reset token counters
    llm.token_counter.reset_counts()
    if hasattr(embed_model, 'token_counter'):
        embed_model.token_counter.reset_counts()

    with open(os.path.join(
            path_to_results, "invalid_llm_outputs.txt"), 'w', encoding='utf-8') as f:
        f.write(str(invalid_llm_outputs))

    has_results = False
    if raw_results != [None]:
        save_results(raw_results=raw_results,
                     path_to_results=path_to_results,
                     first_write=True,
                     results_type='final')
        has_results = True

    return {
        "prompt": llm_single_prompt.query,
        "llm_costs": llm_costs,
        "has_results": has_results
    }


def rearrange_results(results):
    """Rearrange the results."""
    raw_results, invalid_llm_outputs = zip(*results)
    return list(raw_results), list(invalid_llm_outputs)


if __name__ == "__main__":
    # logging.DEBUG for more verbose output, normal logging.INFO, less verbose logging.ERROR
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    config_params = ConfigParams()
    config_params.update_class(
        {
            'evaluation_mode': 'both',  # both, default, precision_recall_f1 or no_evaluation
            # 'gold_standard': 'gist_2025',
            # 'in_sample': 'sample_160',  # 'sample_39', sample_160, sample_20250604
            'filename_list': ['./data/pdfs/sato holdings_2022_report.pdf'],
        })

    experiment_params = ExperimentParams()
    update_dataclass(experiment_params.pipeline_params,
                     {
                         'input_mode': 'text',  # text+table oder text
                         'embed_only': False,  # True or False
                     })

    update_dataclass(experiment_params.semantic_search_params,
                     {
                         'emb_model': "text-embedding-ada-002",  # text-embedding-3-large
                         'context_window': 0,
                     })

    update_dataclass(experiment_params.llm_params,
                     {
                         # gpt-4o-mini-2024-07-18 OR
                         # gpt-oss-120b OR
                         # o3-mini-2025-01-31 OR
                         # gpt-4o-mini-2024-07-18 OR
                         # gpt-35-turbo-16k OR
                         # gpt-4o-2024-11-20 OR
                         # gpt-5-chat-2025-08-07
                         # gpt-4.1-2025-04-14
                         'llm_model': "gpt-4o-mini-2024-07-18",
                         'prompt_type': 'default',  # default oder custom_gaia
                         'year_min': 2013,
                         'year_max': 2024,
                     })

    mlflow_params = MlflowParams(
        mlflow_experiment_path='/Shared/Experiments_prompt_engineering/lisa_test',
    )
    mlflow_params.construct_mlflow_run_name([config_params, experiment_params])
    main(mlflow_params, config_params, experiment_params)
