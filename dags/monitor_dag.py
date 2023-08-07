from datetime import datetime
import pandas as pd
from airflow.decorators import dag, task


@dag(
    schedule="0 10 * * *",
    start_date=datetime(2023, 7, 10),
    # params=default_request,
)
def monitor_dag():
    @task()
    def get_ref_data(**context: dict) -> pd.DataFrame:
        pass
        return 1

    @task()
    def get_live_data() -> pd.DataFrame:
        pass
        return 2

    # get model

    @task()
    def check_data_drift(ref_data, live_data):
        pass

    @task()
    def check_predict_perform(ref_data, live_data):
        pass

    @task()
    def check_predict_drift(ref_data, live_data):
        pass

    @task()
    def register_monitor(d1, d2, d3):
        pass

    ref_data = get_ref_data()
    live_data = get_live_data()

    register_monitor(
        check_data_drift(ref_data, live_data),
        check_predict_perform(ref_data, live_data),
        check_predict_drift(ref_data, live_data),
    )


forecast = monitor_dag()
