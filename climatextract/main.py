"""Main script to run the pipeline."""
import asyncio
from dataclasses import asdict
import json
import logging
import os
import sys
import tomllib
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any

import mlflow

from climatextract.pipeline import FileConfig, ValueRetrieverPipeline, save_results
from climatextract.experiment_setup import Experiment
import climatextract.config as config
from climatextract.params import ConfigParams, ExperimentParams, MlflowParams
from climatextract.evaluator import evaluate
import climatextract.semantic_search as semantic_search
import climatextract.prompts_with_prompt_parsers as prompts_with_prompt_parsers
from climatextract.data_lake_manager import DataLakeManager

from dotenv import load_dotenv
load_dotenv()

def main(
    mlflow_experiment_path: str,
    pdf_input: str | List[str] | None = None,
    gold_standard_path: str | None = None,
    config_path: str = "climxtract.toml"
):
    """
    Run extraction with MLflow tracking.
    
    Starts MLflow first to get run_id, checks evaluation_mode, calls appropriate function
    
    Args:
        mlflow_experiment_path: MLflow experiment path for tracking.
        pdf_input: PDF file(s) to process. If None, uses config.
        gold_standard_path: Path to gold standard. If None, uses config.
        config_path: Path to config file. Defaults to "climxtract.toml".
    """
    # Load config to construct run name and check evaluation_mode
    config_params, experiment_params, _ = _load_config(config_path)
    
    # Set up MLflow
    mlflow_params = MlflowParams(mlflow_experiment_path=mlflow_experiment_path)
    mlflow_params.construct_mlflow_run_name([config_params, experiment_params])
    
    experiment = Experiment(mlflow_params=mlflow_params)
    experiment.setup_experiment()
    mlflow.openai.autolog()

    # Initiate the MLflow run context
    with mlflow.start_run(run_name=mlflow_params.mlflow_run_name) as run:
        run_id = run.info.run_id
        path_to_results = FileConfig.get_path_to_results(run_id=run_id)

        # Check evaluation_mode to decide which function to call
        if config_params.evaluation_mode == "no_evaluation":
            result = _extract_with_metadata(pdf_input, path_to_results, config_path)
        else:
            result = _extract_and_evaluate_with_metadata(
                pdf_input, gold_standard_path, path_to_results, config_path)
        
        if result is None:
            print("Extraction failed or returned no results.")
            return run_id
        
        # Get params from extract/evaluate that were actually used
        config_params = result["config_params"]
        experiment_params = result["experiment_params"]
        
        # Read logs.json created by extract/evaluate, add run_info
        json_log_path = os.path.join(path_to_results, "logs.json")
        with open(json_log_path, 'r', encoding='utf-8') as f:
            json_logs = json.load(f)
        json_logs["run_info"]["run_id"] = run_id
        json_logs["parameters"].update(MlflowParams.filter_params(mlflow_params))

        # Log everything to MLflow
        mlflow.log_params(MlflowParams.filter_params(mlflow_params))
        mlflow.log_params(asdict(config_params))
        mlflow.log_params(asdict(experiment_params.pipeline_params))
        mlflow.log_params(asdict(experiment_params.semantic_search_params))
        mlflow.log_params(asdict(experiment_params.llm_params))
        mlflow.log_param("prompt", result["prompt"])
        mlflow.log_metrics(result["llm_costs"])

        # Log evaluation metrics to MLflow if they exist
        if result.get("evaluation_metrics"):
            mlflow.log_metrics(result["evaluation_metrics"])

        # Save updated logs.json with run_info
        with open(json_log_path, 'w', encoding='utf-8') as f:
            json.dump(json_logs, f, indent=2, ensure_ascii=False, default=str)

        mlflow.log_artifacts(result["path_to_results"])
    
    return run_id


def extract(pdf_input: str | List[str] | None = None) -> Optional[str]:
    """
    Extract CO2 emissions data from PDF reports.
    
    Public API - returns just the path to results.
    
    Args:
        pdf_input: A directory path (processes all PDFs), a single file path,
                   or a list of file paths. If None, uses filename_list from config.
    
    Returns:
        Path to the results directory, or None if extraction failed.
    """
    result = _extract_with_metadata(pdf_input)
    if result is None:
        return None
    return result["path_to_results"]


def extract_and_evaluate(
    pdf_input: str | List[str] | None = None,
    gold_standard_path: str | None = None
) -> Optional[str]:
    """
    Extract CO2 emissions data and evaluate against gold standard.
    
    Public API - always evaluates, regardless of evaluation_mode in config.
    
    Args:
        pdf_input: A directory path (processes all PDFs), a single file path,
                   or a list of file paths. If None, uses filename_list from config.
        gold_standard_path: Path to gold standard dataset. If None, uses config.
    
    Returns:
        Path to the results directory, or None if extraction failed.
    """
    result = _extract_and_evaluate_with_metadata(pdf_input, gold_standard_path)
    if result is None:
        return None
    return result["path_to_results"]


def _extract_with_metadata(pdf_input: str | List[str] | None = None, 
                           path_to_results: str | None = None,
                           config_path: str = "climxtract.toml") -> Optional[Dict[str, Any]]:
    """
    Internal extraction function that returns full metadata for main().
    
    Args:
        pdf_input: A directory path (processes all PDFs), a single file path,
                   or a list of file paths. If None, uses filename_list from config.
        path_to_results: Full path to save results. If None, uses {output_dir}/{uuid}
                         where output_dir comes from config or defaults to "output".
        config_path: Path to config file. Defaults to "climxtract.toml".
    
    Returns:
        Dictionary containing:
        - path_to_results: Where results are saved
        - config_params: ConfigParams that were used
        - experiment_params: ExperimentParams that were used  
        - prompt: The prompt used
        - llm_costs: Token counts and costs
        - has_results: Whether any results were extracted
    """
    # Load config (defaults + config file overrides)
    config_params, experiment_params, output_dir = _load_config(config_path)
    
    # Resolve PDF files: argument takes priority, then config
    if pdf_input is not None:
        pdf_files = _resolve_pdf_input(pdf_input)
    elif config_params.filename_list:
        pdf_files = _resolve_pdf_input(config_params.filename_list)
    else:
        raise ValueError("No PDF files specified in argument or config file.")
    
    # Store actual PDF files used in config_params for logging
    config_params.filename_list = pdf_files
    
    # Determine output path:
    # 1. If path_to_results provided (from main with MLflow): use it
    # 2. Otherwise: {output_dir from config or "output"}/{uuid}
    if path_to_results is None:
        run_id = uuid.uuid4().hex
        path_to_results = os.path.join(output_dir, run_id)
    os.makedirs(path_to_results, exist_ok=True)
    
    # Set up components
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

    # Handle data lake operations with dedicated manager
    storage_account_url = os.environ.get("AZURE_STORAGE_ACCOUNT_URL")
    data_lake_manager = DataLakeManager(storage_account_url)

    if not data_lake_manager.execute_complete_workflow(
        filename_list=pdf_files,
        embeddings_repo=embeddings_repo,
        input_mode=experiment_params.pipeline_params.input_mode
    ):
        print("Data lake workflow failed.")
        return None

    if not search_query.embed_search_query_and_save_to_database(embed_model):
        raise ValueError(
            "Failed to embed the search query and save it to the database.")

    # Build prompt handler
    if experiment_params.llm_params.prompt_type == 'custom_gaia':
        llm_single_prompt = prompts_with_prompt_parsers.CustomPromptGaia(
            prompt_params=experiment_params.llm_params)
    else:
        llm_single_prompt = prompts_with_prompt_parsers.LlmSinglePromptQueryScope12lb2mb3(
            prompt_params=experiment_params.llm_params)

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
            filename_list=pdf_files,
            path_to_results=path_to_results
        )
    )
    if not results:
        print("No pdfs were processed. Exiting.")
        return None

    raw_results, invalid_llm_outputs = _rearrange_results(results)

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

    # Save invalid outputs
    with open(os.path.join(
            path_to_results, "invalid_llm_outputs.txt"), 'w', encoding='utf-8') as f:
        f.write(str(invalid_llm_outputs))

    # Save results
    has_results = False
    if raw_results != [None]:
        save_results(raw_results=raw_results,
                     path_to_results=path_to_results,
                     first_write=True,
                     results_type='final')
        has_results = True

    # Save logs.json with all parameters and metrics (for both public and internal use)
    json_logs = {
        "parameters": {
            **asdict(config_params),
            **asdict(experiment_params.pipeline_params),
            **asdict(experiment_params.semantic_search_params),
            **asdict(experiment_params.llm_params),
            "prompt": llm_single_prompt.query,
        },
        "metrics": llm_costs,
        "run_info": {}
    }
    json_log_path = os.path.join(path_to_results, "logs.json")
    with open(json_log_path, 'w', encoding='utf-8') as f:
        json.dump(json_logs, f, indent=2, ensure_ascii=False, default=str)

    # Return everything that was used
    return {
        "path_to_results": path_to_results,
        "config_params": config_params,
        "experiment_params": experiment_params,
        "prompt": llm_single_prompt.query,
        "llm_costs": llm_costs,
        "has_results": has_results
    }

def _extract_and_evaluate_with_metadata(
    pdf_input: str | List[str] | None = None,
    gold_standard_path: str | None = None,
    path_to_results: str | None = None,
    config_path: str = "climxtract.toml"
) -> Optional[Dict[str, Any]]:
    """
    Internal: extraction + evaluation, returns full metadata.
    
    Args:
        pdf_input: PDF files to process. If None, uses config.
        gold_standard_path: Path to gold standard. If None, uses config.
        path_to_results: Full path to save results. If None, auto-generates.
        config_path: Path to config file. Defaults to "climxtract.toml".
    
    Returns:
        Dictionary with extraction results + evaluation_metrics.
    """

    # Load config (for filenames and eval mode)
    config_params, experiment_params, output_dir = _load_config(config_path)

    # Resolve PDF files: argument takes priority, then config
    if pdf_input is not None:
        pdf_files = _resolve_pdf_input(pdf_input)
    elif config_params.filename_list:
        pdf_files = _resolve_pdf_input(config_params.filename_list)
    else:
        raise ValueError("No PDF files specified in argument or config file.")

    # Determine gold standard: argument > config
    gs_path = gold_standard_path if gold_standard_path else config_params.gold_standard

    # If evaluating, filter PDFs to those present in gold standard (report_name column)
    if config_params.evaluation_mode != "no_evaluation":
        import csv
        gs_report_names = set()
        with open(gs_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            col = "report_name" if "report_name" in reader.fieldnames else None
            if not col:
                raise ValueError(
                    f"Gold standard at '{gs_path}' missing required 'report_name' column"
                )
            for row in reader:
                gs_report_names.add(row[col])
        filtered = [p for p in pdf_files if Path(p).name in gs_report_names]
        if not filtered:
            raise ValueError(
                "No input PDFs match report_name entries in the gold standard."
            )
        pdf_files = filtered

    # Now run extraction with the filtered list, passing through to reuse logic
    result = _extract_with_metadata(pdf_files, path_to_results, config_path)
    if result is None:
        return None
    
    if not result["has_results"]:
        result["evaluation_metrics"] = None
        return result
    
    config_params = result["config_params"]  # updated inside extract
    path = result["path_to_results"]
    
    # Run evaluation
    evaluation_metrics = evaluate(
        path_to_results=path,
        gold_standard=gs_path,
        mode=config_params.evaluation_mode
    )
    
    # Update logs.json with evaluation metrics
    if evaluation_metrics:
        json_log_path = os.path.join(path, "logs.json")
        with open(json_log_path, 'r', encoding='utf-8') as f:
            json_logs = json.load(f)
        json_logs["metrics"].update(evaluation_metrics)
        with open(json_log_path, 'w', encoding='utf-8') as f:
            json.dump(json_logs, f, indent=2, ensure_ascii=False, default=str)
    
    # Add evaluation metrics to result
    result["evaluation_metrics"] = evaluation_metrics
    return result


def _load_config(config_path: str = "climxtract.toml"):
    """Load config from TOML file, with defaults from dataclasses."""
    config_params = ConfigParams()
    experiment_params = ExperimentParams()
    output_dir = "output"  # Default output directory
    
    config_file = Path(config_path)
    if not config_file.exists():
        return config_params, experiment_params, output_dir
    
    with open(config_file, "rb") as f:
        file_config = tomllib.load(f)
    
    # Map sections to dataclass objects
    section_targets = {
        "input": [config_params],
        "evaluation": [config_params],
        "models": [experiment_params.llm_params, experiment_params.semantic_search_params],
        "extraction": [experiment_params.llm_params, experiment_params.pipeline_params, 
                       experiment_params.semantic_search_params],
    }
    
    for section, targets in section_targets.items():
        if section not in file_config:
            continue
        for key, value in file_config[section].items():
            for target in targets:
                if hasattr(target, key):
                    setattr(target, key, value)
                    break
    
    # Get output directory from config
    if "output" in file_config and "output_dir" in file_config["output"]:
        output_dir = file_config["output"]["output_dir"]
    
    return config_params, experiment_params, output_dir


def _resolve_pdf_input(pdf_input: str | List[str]) -> List[str]:
    """Convert pdf_input (directory, file, or list) into a list of file paths."""
    if isinstance(pdf_input, str):
        path = Path(pdf_input)
        if path.is_dir():
            pdf_files = [str(p) for p in path.glob("*.pdf")]
            if not pdf_files:
                raise FileNotFoundError(f"No PDF files found in directory: {pdf_input}")
            return pdf_files
        elif path.is_file():
            return [pdf_input]
        else:
            raise FileNotFoundError(f"Path not found: {pdf_input}")
    else:
        # pdf_input is List[str]
        return pdf_input


def _rearrange_results(results):
    """Rearrange the results."""
    raw_results, invalid_llm_outputs = zip(*results)
    return list(raw_results), list(invalid_llm_outputs)


if __name__ == "__main__":
    # logging.DEBUG for more verbose output, normal logging.INFO, less verbose logging.ERROR
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    # pdf_input is optional - if not specified, uses filename_list from config
    main(
        # pdf_input=['./data/pdfs/sato holdings_2022_report.pdf'], # Optional: overrides config
        # mlflow_experiment_path='/Shared/Experiments_prompt_engineering/lisa_test'
        mlflow_experiment_path='/Shared/Experiments_prompt_engineering/precision_recall_analysis'
    )
