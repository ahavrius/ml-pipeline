import os
from datetime import date, datetime, timedelta
import pandas as pd
import mlflow
from mlflow import MlflowClient
from airflow.decorators import dag, task

from scripts.config.ml_dag_config import default_request, temp_path
from scripts.data_func import (
    load_features,
    save_features,
    load_data,
    write_data,
)
from scripts.model_func import generate_features


@dag(
    schedule="0 9 * * *",
    start_date=datetime(2023, 7, 10),
    params=default_request,
)
def forecast_dag():
    @task()
    def get_data(**context: dict) -> pd.DataFrame:
        yesterday = date.today() - timedelta(days=1)

        time_period = {
            "start_datetime": yesterday.strftime("2018-07-%dT00:00:00.000Z"),
            "end_datetime": yesterday.strftime("2018-07-%dT23:45:00.000Z"),
        }
        updated_time_period = {**time_period, **context["params"]["time_period"]}
        return load_data(context["params"]["input_data"], updated_time_period)

    @task()
    def featurize(data: pd.DataFrame):
        features = generate_features(data)
        save_features(features, None, temp_path())
        return

    @task()
    def fetch_model(**context: dict) -> str:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_REMOTE_TRACKING_URI"))

        client = MlflowClient()
        registered_model = client.get_latest_versions(
            context["params"]["model"]["name"], stages=["None"]
        )
        model_uri = f"models:/{registered_model[0].name}/{registered_model[0].version}"
        return model_uri

    @task()
    def predict(_, model_uri: str) -> list:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_REMOTE_TRACKING_URI"))
        model = mlflow.sklearn.load_model(model_uri)

        features, _ = load_features(temp_path(), is_target=False)
        predict = model.predict(features)
        return predict.tolist()

    @task()
    def upload_predict(predict: list, **context: dict):
        write_data(
            pd.DataFrame(predict, columns=["solar_predict"]),
            **context["params"]["output_data"],
        )

    upload_predict(predict(featurize(get_data()), fetch_model()))


forecast = forecast_dag()
