import pandas as pd
from pathlib import Path


def upload_data(input_data, time_period) -> pd.DataFrame:
    df = read_data(**input_data)
    df = crop_data_span(df, **time_period)

    # data quality check, data cleaning
    return df


def read_data(location_type: str, location_path: str) -> pd.DataFrame:
    return pd.read_parquet(location_path, engine='pyarrow')

def crop_data_span(df: pd.DataFrame,
                   start_datetime: str | None,
                   end_datetime: str | None) -> pd.DataFrame:
    if start_datetime:
        df = df.index >= start_datetime
    if end_datetime:
        df = df.index < end_datetime
    return df

def write_data():
    pass

def save_features(features, target, path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

    features.to_parquet(f"{path}/features.parquet")
    pd.DataFrame(target).to_parquet(f"{path}/target.parquet")

def load_features(path: str):
    features = pd.read_parquet(f"{path}/features.parquet")
    target = pd.read_parquet(f"{path}/target.parquet")
    return features, target