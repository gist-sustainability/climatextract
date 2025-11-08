import json
import os
from src.params import ConfigParams, ExperimentParams, MlflowParams, update_dataclass


def create_test_config(prompt_type, input_mode):
    """Factory for test configurations"""
    mlflow_params = MlflowParams(
        mlflow_experiment_path='/Shared/Experiments_prompt_engineering/acceptance_testing',
        mlflow_run_name=f'{prompt_type}_{input_mode}',
    )

    config_params = ConfigParams()
    config_params.update_class(
        {'filename_list': ['./data/pdfs/sato holdings_2022_report.pdf'],
         # 'filename_list': ['./data/pdfs/addtech_2022_report.pdf'],
         })

    experiment_params = ExperimentParams()
    update_dataclass(experiment_params.pipeline_params,
                     {'input_mode': input_mode})
    update_dataclass(experiment_params.llm_params, {
        'prompt_type': prompt_type,
    })

    return mlflow_params, config_params, experiment_params


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
