import os
import mlflow
from datetime import datetime
import pandas as pd

from airflow.decorators import dag, task

from scripts.config.ml_dag_config import artifact_path, default_request, temp_path
from scripts.data_func import load_features, save_features, load_data
from scripts.model_func import generate_features, generate_target, train_model


def set_experiment(name: str):
    if not mlflow.get_experiment_by_name(name):
        mlflow.create_experiment(name)
    mlflow.set_experiment(name)


@dag(schedule="0 8 * * Mon", start_date=datetime(2023, 7, 10), params=default_request)
def train_dag():
    @task()
    def get_data(**context: dict) -> pd.DataFrame:
        return load_data(
            context["params"]["input_data"], context["params"]["time_period"]
        )

    @task()
    def featurize(data: pd.DataFrame):
        features = generate_features(data)
        target = generate_target(data)

        save_features(features, target, temp_path())
        return

    @task()
    def train(_, **context: dict) -> str:
        run_date = datetime.today().strftime("%Y-%m-%d")

        mlflow.set_tracking_uri(os.getenv("MLFLOW_REMOTE_TRACKING_URI"))
        set_experiment(f"run_{run_date}")

        features, target = load_features(temp_path())
        run = train_model(features, target, artifact_path, context["params"])
        return run.info.run_id

    @task()
    def register_model(ml_run_id: str, **context: dict):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_REMOTE_TRACKING_URI"))
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=ml_run_id, artifact_path=artifact_path
        )
        mlflow.register_model(
            model_uri=model_uri, name=context["params"]["model"]["name"]
        )

    register_model(train(featurize(get_data())))


train = train_dag()
