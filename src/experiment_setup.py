import mlflow
import os

from typing import Dict, List

# This class is responsible for setting up the MLflow experiment


class Experiment:
    """
    This class is responsible for setting up the MLflow experiment.
    """

    def __init__(self, mlflow_params: Dict[str, str]):
        self.mlflow_params = mlflow_params

    def setup_experiment(self) -> List[str]:
        """
        Sets up the MLflow experiment with the given parameters.
        """
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

        self.get_or_create_experiment(
            self.mlflow_params.mlflow_experiment_path)

    def get_or_create_experiment(self, experiment_name):
        """
        This function checks if an experiment with the given name exists in the MLflow server.
        If it exists, it sets this experiment as the active one.
        If it does not exist, it creates a new experiment with the specified name.
        """
        # Check if the experiment already exists
        experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment:
            # Experiment exists, set it as the active experiment
            experiment_id = experiment.experiment_id
            print(
                f"Experiment already exists with ID: {experiment_id}. "
                f"Setting as active experiment."
            )
        else:
            # Experiment does not exist, create it
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"Experiment created with ID: {experiment_id}.")

        # Set the experiment as the active one
        mlflow.set_experiment(experiment_name)
