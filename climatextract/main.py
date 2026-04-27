"""Main module for ClimXtract - CO2 emissions extraction from PDF reports."""
import asyncio
from dataclasses import asdict
import json
import logging
import os
try:
    import tomllib
except ImportError:
    import tomli as tomllib
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any

import mlflow
import nest_asyncio

# Apply nest_asyncio to allow asyncio.run() in environments with running event loops (e.g., Jupyter)
nest_asyncio.apply()

from climatextract.pipeline import FileConfig, ValueRetrieverPipeline, save_results
from climatextract.experiment_setup import Experiment
import climatextract.config as config
from climatextract.params import ConfigParams, ExperimentParams, MlflowParams
from climatextract.evaluator import evaluate
import climatextract.semantic_search as semantic_search
import climatextract.prompts_with_prompt_parsers as prompts_with_prompt_parsers
from climatextract.data_lake_manager import DataLakeManager
from climatextract.console import init_console, get_console

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


def extract(
    pdf_input: str | List[str] | None = None,
    config_path: str = "climatextract.toml",
    enable_mlflow: bool = False,
    verbose: bool = False
) -> Optional[str]:
    """
    Extract CO2 emissions data from PDF reports.

    Public API - returns just the path to results.

    Args:
        pdf_input: A directory path (processes all PDFs), a single file path,
                   or a list of file paths. If None, uses filename_list from config.
        config_path: Path to config file. Defaults to "climatextract.toml".
        enable_mlflow: Whether to log results to MLflow. If True, uses MLflow settings
                       from .env (tracking_uri) and config file (experiment_name). Enables
                       full MLflow tracking including OpenAI autolog and traces.
        verbose: Whether to show detailed per-PDF output. Defaults to False.

    Returns:
        Path to the results directory, or None if extraction failed.
    """
    # Initialize console
    console = init_console(verbose=verbose)

    if enable_mlflow:
        # Load config to get MLflow settings
        config_params, experiment_params, output_dir, mlflow_config, _ = _load_config(config_path)

        # Set up MLflow with proper run name
        mlflow_params = MlflowParams(mlflow_experiment_path=mlflow_config["experiment_name"])
        mlflow_params.construct_mlflow_run_name([config_params, experiment_params])

        experiment = Experiment(mlflow_params=mlflow_params, tracking_uri=mlflow_config["tracking_uri"])
        experiment.setup_experiment()
        mlflow.openai.autolog()  # Capture all OpenAI API calls

        with mlflow.start_run(run_name=mlflow_params.mlflow_run_name) as run:
            run_id = run.info.run_id
            path_to_results = FileConfig.get_path_to_results(run_id=run_id, output_dir=output_dir)

            # Run extraction inside MLflow context
            result = _extract_with_metadata(pdf_input, path_to_results, config_path)
            if result is None:
                return None

            # Get params from extraction that were actually used
            config_params = result["config_params"]
            experiment_params = result["experiment_params"]

            # Update config.json with run_id and MLflow params
            json_log_path = os.path.join(path_to_results, "config.json")
            with open(json_log_path, 'r', encoding='utf-8') as f:
                json_logs = json.load(f)
            json_logs["run_info"]["run_id"] = run_id
            json_logs["parameters"].update(MlflowParams.filter_params(mlflow_params))
            with open(json_log_path, 'w', encoding='utf-8') as f:
                json.dump(json_logs, f, indent=2, ensure_ascii=False, default=str)

            # Log everything to MLflow
            mlflow.log_params(MlflowParams.filter_params(mlflow_params))
            mlflow.log_params(asdict(config_params))
            mlflow.log_params(asdict(experiment_params.pipeline_params))
            mlflow.log_params(asdict(experiment_params.semantic_search_params))
            mlflow.log_params(asdict(experiment_params.llm_params))
            mlflow.log_param("prompt", result["prompt"])
            mlflow.log_metrics(result["llm_costs"])
            mlflow.log_artifacts(path_to_results)

            # Print results (top line + path inside `with` block)
            console.print_results_start(path_to_results)

        # Bottom line after `with` block (MLflow prints 🏃/🧪 URLs during __exit__)
        console.print_results_end()
        return path_to_results
    else:
        # No MLflow - simple extraction
        result = _extract_with_metadata(pdf_input, config_path=config_path)
        if result is None:
            return None

        # Print results
        console.print_results(result["path_to_results"])

        return result["path_to_results"]


def extract_and_evaluate(
    pdf_input: str | List[str] | None = None,
    gold_standard_path: str | None = None,
    config_path: str = "climatextract.toml",
    enable_mlflow: bool = False,
    verbose: bool = False
) -> Optional[str]:
    """
    Extract CO2 emissions data and evaluate against gold standard.

    Public API - always evaluates, regardless of evaluation_mode in config.

    Args:
        pdf_input: A directory path (processes all PDFs), a single file path,
                   or a list of file paths. If None, uses filename_list from config.
        gold_standard_path: Path to gold standard dataset. If None, uses config.
        config_path: Path to config file. Defaults to "climatextract.toml".
        enable_mlflow: Whether to log results to MLflow. If True, uses MLflow settings
                       from .env (tracking_uri) and config file (experiment_name). Enables
                       full MLflow tracking including OpenAI autolog and traces.
        verbose: Whether to show detailed per-PDF output. Defaults to False.

    Returns:
        Path to the results directory, or None if extraction failed.
    """
    # Initialize console
    console = init_console(verbose=verbose)

    if enable_mlflow:
        # Load config to get MLflow settings
        config_params, experiment_params, output_dir, mlflow_config, _ = _load_config(config_path)

        # Set up MLflow with proper run name
        mlflow_params = MlflowParams(mlflow_experiment_path=mlflow_config["experiment_name"])
        mlflow_params.construct_mlflow_run_name([config_params, experiment_params])

        experiment = Experiment(mlflow_params=mlflow_params, tracking_uri=mlflow_config["tracking_uri"])
        experiment.setup_experiment()
        mlflow.openai.autolog()  # Capture all OpenAI API calls

        with mlflow.start_run(run_name=mlflow_params.mlflow_run_name) as run:
            run_id = run.info.run_id
            path_to_results = FileConfig.get_path_to_results(run_id=run_id, output_dir=output_dir)

            # Run extraction+evaluation inside MLflow context
            result = _extract_and_evaluate_with_metadata(
                pdf_input, gold_standard_path, path_to_results, config_path)
            if result is None:
                return None

            # Get params from extraction that were actually used
            config_params = result["config_params"]
            experiment_params = result["experiment_params"]

            # Update config_and_metrics.json with run_id and MLflow params
            json_log_path = os.path.join(path_to_results, "config_and_metrics.json")
            with open(json_log_path, 'r', encoding='utf-8') as f:
                json_logs = json.load(f)
            json_logs["run_info"]["run_id"] = run_id
            json_logs["parameters"].update(MlflowParams.filter_params(mlflow_params))
            with open(json_log_path, 'w', encoding='utf-8') as f:
                json.dump(json_logs, f, indent=2, ensure_ascii=False, default=str)

            # Log everything to MLflow
            mlflow.log_params(MlflowParams.filter_params(mlflow_params))
            mlflow.log_params(asdict(config_params))
            mlflow.log_params(asdict(experiment_params.pipeline_params))
            mlflow.log_params(asdict(experiment_params.semantic_search_params))
            mlflow.log_params(asdict(experiment_params.llm_params))
            mlflow.log_param("prompt", result["prompt"])
            mlflow.log_metrics(result["llm_costs"])

            # Log evaluation metrics if they exist
            if result.get("evaluation_metrics"):
                mlflow.log_metrics(result["evaluation_metrics"])

            mlflow.log_artifacts(path_to_results)

            # Print results (top line + path inside `with` block)
            console.print_results_start(path_to_results)

        # Bottom line after `with` block (MLflow prints 🏃/🧪 URLs during __exit__)
        console.print_results_end()
        return path_to_results
    else:
        # No MLflow - simple extraction + evaluation
        result = _extract_and_evaluate_with_metadata(pdf_input, gold_standard_path, config_path=config_path)
        if result is None:
            return None

        # Print results
        console.print_results(result["path_to_results"])

        return result["path_to_results"]


def _extract_with_metadata(pdf_input: str | List[str] | None = None,
                           path_to_results: str | None = None,
                           config_path: str = "climatextract.toml") -> Optional[Dict[str, Any]]:
    """
    Internal extraction function that returns full metadata for main().

    Args:
        pdf_input: A directory path (processes all PDFs), a single file path,
                   or a list of file paths. If None, uses filename_list from config.
        path_to_results: Full path to save results. If None, uses {output_dir}/{uuid}
                         where output_dir comes from config or defaults to "output".
        config_path: Path to config file. Defaults to "climatextract.toml".

    Returns:
        Dictionary containing:
        - path_to_results: Where results are saved
        - config_params: ConfigParams that were used
        - experiment_params: ExperimentParams that were used
        - prompt: The prompt used
        - llm_costs: Token counts and costs
        - has_results: Whether any results were extracted
    """
    console = get_console()

    # Load config (defaults + config file overrides)
    config_params, experiment_params, output_dir, _, datalake_config = _load_config(config_path)

    # Handle data lake operations with dedicated manager
    storage_account_url = os.environ.get("AZURE_STORAGE_ACCOUNT_URL")
    data_lake_manager = DataLakeManager(
        storage_account_url,
        blob_path_pdfs=datalake_config["blob_path_pdfs"],
        blob_path_embeddings=datalake_config["blob_path_embeddings"],
    )

    # If input is a directory (existing but empty, or not yet created), try populating from data lake
    raw_input = pdf_input if pdf_input is not None else config_params.filename_list
    if isinstance(raw_input, str) and not raw_input.endswith(".pdf"):
        input_path = Path(raw_input)
        is_empty_dir = input_path.is_dir() and not any(input_path.glob("*.pdf"))
        is_missing_dir = not input_path.exists()
        if is_empty_dir or is_missing_dir:
            data_lake_manager.download_directory_from_blob(raw_input)

    # Resolve PDF files: argument takes priority, then config
    if pdf_input is not None:
        pdf_files = _resolve_pdf_input(pdf_input)
    elif config_params.filename_list:
        pdf_files = _resolve_pdf_input(config_params.filename_list)
    else:
        raise ValueError("No PDF files specified in argument or config file.")

    # Check for missing PDFs upfront and offer to download
    data_lake_manager.download_pdfs_if_not_locally_available(pdf_files)

    # Store actual PDF files used in config_params for logging
    config_params.filename_list = pdf_files

    # Print header and config
    console.print_header()
    console.print_config(
        llm_model=experiment_params.llm_params.llm_model,
        embedding_model=experiment_params.semantic_search_params.emb_model,
        input_mode=experiment_params.pipeline_params.input_mode,
        pdf_count=len(pdf_files)
    )

    # Determine output path:
    # 1. If path_to_results provided (from main with MLflow): use it
    # 2. Otherwise: {output_dir from config or "output"}/{uuid}
    if path_to_results is None:
        run_id = uuid.uuid4().hex
        path_to_results = os.path.join(output_dir, run_id)
    os.makedirs(path_to_results, exist_ok=True)

    # Set up components
    # Use custom embeddings repository if specified, otherwise fall back to default
    if experiment_params.semantic_search_params.embeddings_repository:
        embeddings_repo = semantic_search.EmbeddingsRepository(
            database_name=experiment_params.semantic_search_params.embeddings_repository
        )
    else:
        embeddings_repo = semantic_search.EmbeddingsRepository(
            database_name=(
                f"data/processed/embeddings/"
                f"{experiment_params.semantic_search_params.emb_model}"
                f"_from_2025_12_23.duckdb")
        )

    embed_model = config.EmbeddingModel(
        model_name=experiment_params.semantic_search_params.emb_model)
    llm = config.Llm(
        model_name=experiment_params.llm_params.llm_model,
        return_logprobs=experiment_params.llm_params.return_logprobs,
        max_parallel_llm_prompts_running=experiment_params.llm_params.max_parallel_llm_prompts_running,
        print_query_duration=False
    )

    search_query = semantic_search.SearchQuery(
        search_query=experiment_params.semantic_search_params.search_query,
        repository=embeddings_repo)

    # Check database and file status for console output
    database_exists = embeddings_repo.database_exists()
    search_query_cached = False
    if database_exists:
        search_query_cached = embeddings_repo.search_query_exists(
            experiment_params.semantic_search_params.search_query)

    # Check which PDFs need embedding
    pdfs_needing_embedding = 0
    pdf_cache_status = {}
    if database_exists:
        for filepath in pdf_files:
            short_filename = os.path.basename(filepath)
            cached = embeddings_repo.pdf_exists(short_filename)
            pdf_cache_status[short_filename] = cached
            if not cached:
                pdfs_needing_embedding += 1
    else:
        pdfs_needing_embedding = len(pdf_files)
        for filepath in pdf_files:
            short_filename = os.path.basename(filepath)
            pdf_cache_status[short_filename] = False

    # Print setup info
    console.print_setup(
        database_name=embeddings_repo.get_database_name(),
        database_exists=database_exists,
        pdfs_needing_embedding=pdfs_needing_embedding,
        total_pdfs=len(pdf_files),
        search_query_cached=search_query_cached,
        pdf_cache_status=pdf_cache_status if console.verbose else None
    )

    if not data_lake_manager.execute_complete_workflow(
        filename_list=pdf_files,
        embeddings_repo=embeddings_repo,
        input_mode=experiment_params.pipeline_params.input_mode
    ):
        logger.warning("Data lake workflow failed.")
        return None

    if not search_query.embed_search_query_and_save_to_database(embed_model):
        raise ValueError(
            "Failed to embed the search query and save it to the database.")

    # Build prompt handler
    if experiment_params.llm_params.prompt_type == 'structured_json':
        llm_single_prompt = prompts_with_prompt_parsers.StructuredJsonPrompt(
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
        llm_single_prompt=llm_single_prompt,
        console=console
    )

    # Start embedding progress (if any PDFs need embedding)
    console.start_embedding_progress(pdfs_needing_embedding)

    # Start extraction progress
    console.start_extraction_progress(len(pdf_files))

    try:
        results = asyncio.run(
            retriever_pipeline.retrieve_values_for_doc_list(
                filename_list=pdf_files,
                path_to_results=path_to_results
            )
        )
    finally:
        console.stop_embedding_progress()
        console.stop_extraction_progress()

    if not results:
        logger.warning("No PDFs were processed. Exiting.")
        return None

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

    # Save results
    has_results = False
    if results != [None]:
        save_results(raw_results=results,
                     path_to_results=path_to_results,
                     first_write=True,
                     results_type='final',
                     console=console)
        has_results = True

    # Save config.json with all parameters and metrics (for both public and internal use)
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
    json_log_path = os.path.join(path_to_results, "config.json")
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
    config_path: str = "climatextract.toml",
) -> Optional[Dict[str, Any]]:
    """
    Internal: extraction + evaluation, returns full metadata.

    Args:
        pdf_input: PDF files to process. If None, uses config.
        gold_standard_path: Path to gold standard. If None, uses config.
        path_to_results: Full path to save results. If None, auto-generates.
        config_path: Path to config file. Defaults to "climatextract.toml".

    Returns:
        Dictionary with extraction results + evaluation_metrics.
    """
    console = get_console()

    # Load config (for filenames and eval mode)
    config_params, experiment_params, output_dir, _, datalake_config = _load_config(config_path)

    # If input is an empty directory, try populating it from the data lake
    storage_account_url = os.environ.get("AZURE_STORAGE_ACCOUNT_URL")
    data_lake_manager = DataLakeManager(
        storage_account_url,
        blob_path_pdfs=datalake_config["blob_path_pdfs"],
        blob_path_embeddings=datalake_config["blob_path_embeddings"],
    )
    raw_input = pdf_input if pdf_input is not None else config_params.filename_list
    if isinstance(raw_input, str) and not raw_input.endswith(".pdf"):
        input_path = Path(raw_input)
        is_empty_dir = input_path.is_dir() and not any(input_path.glob("*.pdf"))
        is_missing_dir = not input_path.exists()
        if is_empty_dir or is_missing_dir:
            data_lake_manager.download_directory_from_blob(raw_input)

    # Resolve PDF files: argument takes priority, then config
    if pdf_input is not None:
        pdf_files = _resolve_pdf_input(pdf_input)
    elif config_params.filename_list:
        pdf_files = _resolve_pdf_input(config_params.filename_list)
    else:
        raise ValueError("No PDF files specified in argument or config file.")

    # Check for missing PDFs upfront and offer to download
    data_lake_manager.download_pdfs_if_not_locally_available(pdf_files)

    # Determine gold standard: argument > config
    gs_path = gold_standard_path if gold_standard_path else config_params.gold_standard

    # If evaluating, filter PDFs to those present in gold standard (report_name column)
    gold_standard_count = 0
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
        gold_standard_count = len(gs_report_names)
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

    # Print evaluation results
    if evaluation_metrics:
        console.print_evaluation(
            metrics=evaluation_metrics,
            processed_count=len(pdf_files),
            gold_standard_count=gold_standard_count
        )

    # Update config_and_metrics.json with evaluation metrics
    if evaluation_metrics:
        json_log_path = os.path.join(path, "config_and_metrics.json")
        with open(json_log_path, 'r', encoding='utf-8') as f:
            json_logs = json.load(f)
        json_logs["metrics"].update(evaluation_metrics)
        with open(json_log_path, 'w', encoding='utf-8') as f:
            json.dump(json_logs, f, indent=2, ensure_ascii=False, default=str)

    # Add evaluation metrics to result
    result["evaluation_metrics"] = evaluation_metrics
    return result


def _load_config(config_path: str = "climatextract.toml"):
    """Load config from TOML file, with defaults from dataclasses.

    Returns:
        Tuple of (config_params, experiment_params, output_dir, mlflow_config)
    """
    config_params = ConfigParams()
    experiment_params = ExperimentParams()
    output_dir = "output"  # Default output directory
    mlflow_config = {
        "tracking_uri": os.environ.get("MLFLOW_TRACKING_URI", "./mlruns"),
        "experiment_name": "climatextract_experiments"
    }
    datalake_config = {
        "blob_path_pdfs": "pdfs",
        "blob_path_embeddings": "embeddings",
    }

    config_file = Path(config_path)
    if not config_file.exists():
        return config_params, experiment_params, output_dir, mlflow_config, datalake_config

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

    # Get MLflow experiment name from config (tracking_uri comes from .env)
    if "mlflow" in file_config:
        if "experiment_name" in file_config["mlflow"]:
            mlflow_config["experiment_name"] = file_config["mlflow"]["experiment_name"]

    # Get datalake settings from config (with fallbacks)
    if "datalake" in file_config:
        for key in datalake_config:
            if key in file_config["datalake"]:
                datalake_config[key] = file_config["datalake"][key]

    return config_params, experiment_params, output_dir, mlflow_config, datalake_config


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
            response = input("File " + pdf_input + " not found locally. Errors will occur if the file is not already available in the embedding database and not downloadable from datalake. Continue anyway? (y/n): ").strip().lower()
            if response not in ['y', 'yes']:
                raise FileNotFoundError(f"File not found: {pdf_input}")
            return [pdf_input]
    else:
        # pdf_input is List[str]
        return pdf_input