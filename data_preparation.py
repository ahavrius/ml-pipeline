import pandas as pd

# from serve_type import type

def upload_data(request) -> pd.DataFrame:
    df = read_data(**request.input_data)
    df = crop_data_span(df, **request.time_period)

    # data quality check, data cleaning
    return df


def read_data(location_type: str, location_path: str) -> pd.DataFrame:
    return pd.read_parquet(location_path)


def crop_data_span(df: pd.DataFrame,
                   start_datetime: str | None,
                   end_datetime: str | None) -> pd.DataFrame:
    if not start_datetime:
        start_datetime = df["start_of_quarter"][0]
    if not end_datetime:
        end_datetime = df["start_of_quarter"][-1]

    return df[df["start_of_quarter"] >= start_datetime and df["start_of_quarter"] < end_datetime]

def write_data():
    pass
