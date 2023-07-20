import os
from datetime import datetime


def temp_path():
    run_date = datetime.today().strftime("%Y-%m-%d")
    return f"predict_artifacts/{run_date}"


artifact_path = os.getenv("MLFLOW_ARTIFACT_PATH")

default_request = {
    "input_data": {
        "location_path": "gs://dexter-public/solar-dataset.pq",
        "location_type": "local",
    },
    "time_period": {"start_datetime": None, "end_datetime": None},
    "model": {"name": "RandomForestCustom", "param": {}},
    "output_data": {
        "location_path": "data/solar-predict.parquet",
        "location_type": "local",
    },
}

