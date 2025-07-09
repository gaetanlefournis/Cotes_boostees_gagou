"""
MLFlow Integration Module
This module provides a simple interface to interact with MLFlow for :
- logging parameters, 
- logging metrics,
- logging artifacts.
"""
from typing import Union

import mlflow
from mlflow import MlflowClient


class MLFlow():
    """
    A class to manage MLflow runs, logging parameters, metrics, and artifacts.
    """
    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        run_name: str,
        **kwargs,
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name

        self.already_exists = False
        self.experiment_id = None
        self.client = None
        self.run = None
        self.run_id = None

        # Initialize the MLFlow client and set the tracking URI
        self._instantiate()

    def _instantiate(self) -> None:
        """
        Initialize the MLFlow client and set the tracking URI.
        """
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        self.client = MlflowClient()
        self.experiment_id = self.client.get_experiment_by_name(self.experiment_name).experiment_id

    def _run_already_exists(self) -> int:
        """
        Check if a run with the same name already exists.
        If it does, it returns True
        """
        existing_runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"tags.mlflow.runName = '{self.run_name}'",
        )

        if existing_runs:
            print(f"✅ Found existing run with name '{self.run_name}'.")
            self.run = existing_runs[0]
            self.run_id = self.run.info.run_id
            return self.run_id

    def start_run(self) -> None:
        """
        Start a new MLflow run.
        If a run with the same name already exists, it will not create a new one.
        """
        if self._run_already_exists():
            self.already_exists = True
            print("Using existing run.")
            return

        print("🚀 No existing run found. Starting a run now...")
        self.run = mlflow.start_run(run_name=self.run_name)
        self.run_id = self.run.info.run_id
        print(f"🚀 Started MLflow run with ID: {self.run_id}")

    def log_params(self, params: dict):
        """
        Log parameters to the current MLflow run.
        :param params: A dictionary of parameters to log.
        """
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: dict, step: int = None):
        """
        Log metrics to the current MLflow run.
        :param metrics: A dictionary of metrics to log.
        :param step: An optional step value for the metrics.
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_artifacts(self, path: Union[str, list[str]]):
        """
        Log an artifact to the current MLflow run.
        :param path: The path to the artifact file or directory.
        """
        if isinstance(path, list):
            for p in path:
                mlflow.log_artifact(p)
        else:
            mlflow.log_artifact(path)

    def log_text(self, text: str, artifact_file: str):
        """
        Log a text file to the current MLflow run.
        :param text: The text content to log.
        :param artifact_file: The name of the artifact file to create.
        """
        with open(artifact_file, "w", encoding="utf-8") as f:
            f.write(text)
        mlflow.log_artifact(artifact_file)

    def log_model(self, model, model_name: str):
        """
        Log a model to the current MLflow run.
        :param model: The model to log.
        :param model_name: The name to give the logged model.
        """
        mlflow.pytorch.log_model(model, model_name)

    def load_model(self, model_name: str) -> Union[None, object]:
        """
        Load a model from the MLflow registry.
        :return: The loaded model or None if not found.
        """
        try:
            if self._run_already_exists():
                # If the run already exists, find the run_id and load the model from the existing run
                model_uri = f"runs:/{self.run_id}/{model_name}"
                print(model_uri)
                print(f"🔍 Loading model from MLflow run ID: {self.run_id}")
                model = mlflow.pytorch.load_model(model_uri)
                print("✅ Model loaded successfully.")
                return model
            print("❌ No existing run found. Cannot load model.")
            return None
        except Exception as e:
            print(f"❌ Failed to load model from MLflow: {e}")
            return None

    def end_run(self):
        """
        End the current MLflow run.
        """
        mlflow.end_run()
        print("🛑 Ended MLflow run.")
