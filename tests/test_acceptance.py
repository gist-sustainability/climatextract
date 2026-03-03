import logging
import pytest
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

from tests.helpers_testing import save_run_id, get_run_id, create_test_config, cleanup_test_config
from climatextract.main import main


# Test general functionality
@pytest.mark.parametrize("prompt_type,input_mode", [
    ("default", "text"),
    ("custom_gaia", "text"),
    ("default", "text+table"),
    ("custom_gaia", "text+table")
])
def test_functionality(prompt_type, input_mode):
    """
    Test the main functionality with different prompt types and input modes.

    This test verifies that:
    1. The main function runs successfully and returns a valid run_id.
    2. The MLflow experiment status is 'FINISHED'.
    3. Artifacts are generated and contain at least one row of data.

    Parameters:
    - prompt_type: The type of prompt to use ('default' or 'custom_gaia').
    - input_mode: The input mode to use ('text' or 'text+table').
    """
    # Configuration
    mlflow_experiment_path, config_path = create_test_config(prompt_type, input_mode)

    try:
        # Execution with new main() signature
        run_id = main(
            mlflow_experiment_path=mlflow_experiment_path,
            config_path=config_path
        )
        save_run_id(run_id, prompt_type, input_mode)
    finally:
        # Cleanup test config file
        cleanup_test_config(config_path)

    # Connection to MLflow
    client = MlflowClient()
    run_details = client.get_run(run_id)

    # Status check
    assert run_details.info.status == 'FINISHED', f"Run status is {run_details.info.status}, expected 'FINISHED'"

    # Artifact validation
    artifacts = mlflow.artifacts.list_artifacts(run_id=run_id)
    assert artifacts, f"No artifacts found for the run {run_id}"

    target_artifact = next((artifact for artifact in artifacts if artifact.path.endswith(
        "03_co2_emission_table2_w_query_responses.csv")), None)
    if target_artifact:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=target_artifact.path)
        df_artifacts = pd.read_csv(local_path)
    else:
        df_artifacts = None
        print("The specified artifact was not found.")

    assert len(
        df_artifacts) > 0, "Artifact file is empty, expected at least one row."

    print(
        f"Test completed successfully for run_id: {run_id} with prompt_type: {prompt_type} and input_mode: {input_mode}")


# Test quality of results
@pytest.mark.parametrize("prompt_type,input_mode", [
    ("default", "text"),
    ("custom_gaia", "text"),
    ("default", "text+table"),
    ("custom_gaia", "text+table")
])
def test_quality(prompt_type, input_mode):
    """
        Test the quality of results for different prompt types and input modes.

        This test verifies that:
        1. The required artifact '04a_results_available_in_report.csv' is present.
        2. The 'value_match' column contains the expected number of True and NA values.
        3. Any mismatches in 'value_match' are reported.
        4. Optionally, the number of True values in 'unit_match' and mismatches are reported.

        Parameters:
        - prompt_type: The type of prompt to use ('default' or 'custom_gaia').
        - input_mode: The input mode to use ('text' or 'text+table').
        """
    # Configuration
    expected_artifact_name = "04a_results_available_in_report.csv"
    expected_true_value_matches = 32

    # Mlflow setup
    run_id = get_run_id(prompt_type, input_mode)
    if run_id is None:
        pytest.skip("run_id is None, skipping the test")

    artifacts = mlflow.artifacts.list_artifacts(run_id=run_id)
    assert artifacts, f"No artifacts found for run {run_id}"

    artifact_path = next(
        (artifact.path for artifact in artifacts if artifact.path == expected_artifact_name), None)
    if not artifact_path:
        pytest.fail(
            f"Expected artifact '{expected_artifact_name}' not found in run {run_id}")

    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path)
    df_artifacts = pd.read_csv(local_path)

    # Quality checks
    filtered_df = df_artifacts[df_artifacts['value_match'].notna()]
    counts_vm = filtered_df['value_match'].value_counts()
    counts_um = filtered_df['unit_match'].value_counts()

    if False in counts_vm.index:
        false_rows = filtered_df[filtered_df['value_match'] == False]
        false_df = false_rows[['ReportName', 'scope_man', 'extracted_scope_from_llm', 'year_man',
                               'extracted_year_from_llm', 'value_man', 'extracted_value_from_llm',  'unit_man', 'extracted_unit_from_llm', 'page_number_used_by_llm', 'value_match', 'unit_match']]
        logging.info(
            f"{len(false_df)} mismatches found:\n{false_df.to_string(index=False)}")

    if True in counts_vm.index:
        assert counts_vm[
            True] == expected_true_value_matches, f"Expected {expected_true_value_matches} True values in 'value_match', got {counts_vm[True]}"
        logging.info(f"True values in 'unit_match': {counts_um.get(True, 0)}")
        assert df_artifacts['value_match'].isna().sum(
        ) == 8, f"Expected 8 NaN values in 'value_match', got {df_artifacts['value_match'].isna().sum()}"
    else:
        pytest.fail("No True values found in 'value_match'.")
