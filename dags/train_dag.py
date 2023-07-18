from airflow.decorators import dag, task
from datetime import datetime
import pandas as pd
import pickle

from scripts.data_preparation import upload_data, save_features, load_features
from scripts.model_core import generate_features, generate_target, get_power_forecaster, compute_rmse

todays_date = datetime.today().strftime('%Y/%m/%d')
training_output_path = f"training_output/{todays_date}"


request = {
    "input_data": {
        "location_path": "data/solar-dataset.parquet",
        "location_type": "local"
    },
    "time_period": {
        "start_datetime": None,
        "end_datetime": None
    }
}


@dag(schedule='@daily', start_date=datetime(2023, 7, 10))
def train_dag():

    @task()
    def get_data():
        return upload_data(**request)

    @task()
    def featurize(data):
        features = generate_features(data)
        target = generate_target(data)

        save_features(features, target, training_output_path)
        return "fff"

    @task()
    def train_model(arg):
        features, target = load_features(training_output_path)
        model = get_power_forecaster()
        model.fit(X=features, y=target)
        predict = model.predict(features)

        with open(f"{training_output_path}/model.pkl", 'wb') as file:
            pickle.dump(model, file)
        # save predict ?
        return predict
    
    @task()
    def calculate_metrics(predict):
        target = pd.read_parquet(f"{training_output_path}/target.parquet")

        rmse = compute_rmse(target.to_numpy(), predict)
        # save score ?
        print(rmse)
    
    calculate_metrics(train_model(featurize(get_data())))
    # calculate_metrics(model_output["target"], model_output["predict"])

train = train_dag()    