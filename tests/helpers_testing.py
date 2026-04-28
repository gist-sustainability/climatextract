import json
import os


def create_test_config(prompt_type, input_mode):
    """
    Create a test config file and return the config path.

    Returns:
        str: config_path
    """
    # Create test config content
    # Leave llm_model, context_window, year_min, year_max etc to use defaults
    config_content = f'''# Test configuration for {prompt_type}_{input_mode}

[input]
filename_list = ["./data/pdfs/test_report/report.pdf"]

[models]
# Pinned so the quality test's expected match counts stay reproducible
# regardless of changes to the adapter's default model.
llm_model = "gpt-4o"

[extraction]
input_mode = "{input_mode}"
prompt_type = "{prompt_type}"
embeddings_repository = "data/processed/embeddings/test_embeddings/test_embeddings.duckdb"

[evaluation]
gold_standard = "./data/evaluation_dataset/test_evaluation/gold_standard.csv"

[mlflow]
experiment_name = "/Shared/Experiments_prompt_engineering/acceptance_testing"
'''

    # Write to a temp config file
    config_path = f'tests/test_config_{prompt_type}_{input_mode}.toml'
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)

    return config_path


def cleanup_test_config(config_path):
    """Remove test config file after test."""
    if os.path.exists(config_path):
        os.remove(config_path)


def save_run_id(run_id, prompt_type, input_mode):
    file_path = 'tests/last_run_ids.json'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    new_entry = {
        "run_id": run_id,
        "prompt_type": prompt_type,
        "input_mode": input_mode
    }

    # Read existing data or create an empty list
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    else:
        data = []

    # Update existing entry or append new one
    updated = False
    for entry in data:
        if entry['prompt_type'] == prompt_type and entry['input_mode'] == input_mode:
            entry['run_id'] = run_id
            updated = True
            break

    if not updated:
        data.append(new_entry)

    # Write updated data back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')


def get_run_id(prompt_type, input_mode):
    file_path = 'tests/last_run_ids.json'
    if not os.path.exists(file_path):
        open(file_path, 'w', encoding='utf-8').close()
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            if entry['prompt_type'] == prompt_type and entry['input_mode'] == input_mode:
                return entry['run_id']
    return None
