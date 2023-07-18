from datetime import datetime
import mlflow
import os

from mlflow import MlflowClient
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from airflow.decorators import dag, task

from scripts.data_preparation import upload_data, save_features, load_features
from scripts.model_core import generate_features, generate_target, train_model

def run_date():
    return datetime.today().strftime('%Y-%m-%d')

temp_path = f"predict_artifacts/{run_date()}"
artifact_path = os.getenv("MLFLOW_ARTIFACT_PATH")

default_request = {
    "input_data": {
        "location_path": "gs://dexter-public/solar-dataset.pq",
        "location_type": "local"
    },
    "time_period": {
        "start_datetime": None,
        "end_datetime": None
    },
    "output_data": {
        "location_path": "predict/solar-dataset.parquet",
        "location_type": "local"
    },
}

model_name = "RandomForest"

@dag(schedule='@daily',
     start_date=datetime(2023, 7, 10),
     params=default_request)
def train_dag():

    @task()
    def get_data(**context):
        return upload_data(context["params"]["input_data"], context["params"]["time_period"])

    @task()
    def featurize(data):
        features = generate_features(data)
        target = generate_target(data)

        save_features(features, target, temp_path)
        return

    @task()
    def train(arg):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_REMOTE_TRACKING_URI"))

        if not mlflow.get_experiment_by_name(f"run_{run_date()}"):
            mlflow.create_experiment(f"run_{run_date()}")
        mlflow.set_experiment(f"run_{run_date()}")

        features, target = load_features(temp_path)
        run = train_model(features, target, artifact_path)
        return run.info.run_id
    
    @task()
    def register_model(ml_run_id):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_REMOTE_TRACKING_URI"))

        client = MlflowClient()
        # client.create_registered_model(model_name)

        model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=ml_run_id, artifact_path=artifact_path)
        # model_src = RunsArtifactRepository.get_underlying_uri(runs_uri)
        # mv = client.create_model_version(model_name, model_src, ml_run_id)
 
        mv = mlflow.register_model(model_uri=model_uri, name=model_name)

        print("Name: {}".format(mv.name))
        print("Version: {}".format(mv.version))
        print("Description: {}".format(mv.description))
        print("Status: {}".format(mv.status))
        print("Stage: {}".format(mv.current_stage))
    
    register_model(train(featurize(get_data())))

train = train_dag()    