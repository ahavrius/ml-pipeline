import numpy as np
import pandas as pd
import yaml
import mlflow
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def train_model(
    features: pd.DataFrame, target: pd.Series, artifact_path: str, param: dict
) -> mlflow.ActiveRun:
    
    # update model parameters
    with open("scripts/config/model_config.yml", "r") as file:
        default_param = yaml.safe_load(file)
    model_param = {**default_param["random_forest"], **param["model"]["param"]}

    with mlflow.start_run() as run:
        # model creation
        model = get_power_forecaster(model_param)
        model.fit(X=features, y=target)
        predict = model.predict(features)
        rmse = compute_rmse(target.to_numpy(), predict)

        # model logging
        signature = infer_signature(features, predict)
        dataset = mlflow.data.from_pandas(
            features,
            source=param["input_data"]["location_path"]
        )
        mlflow.log_input(dataset, context="training")
        mlflow.log_params(model_param)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(
            model, artifact_path=artifact_path, signature=signature
        )
    return run


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[["cloudcover_low", "cloudcover_mid", "cloudcover_high"]].assign(
        hour=lambda df: df.index.hour.astype("float"),
        month=lambda df: df.index.month.astype("float"),
        minutes=lambda df: 60.0 * df.index.hour + df.index.minute,
        day=lambda df: df.index.dayofyear.astype("float"),
    )


def generate_target(df: pd.DataFrame) -> pd.Series | None:
    column_name = "power"
    if column_name in df.columns:
        return df[column_name]
    return None


def get_power_forecaster(param: dict) -> RandomForestRegressor:
    return RandomForestRegressor(**param)


def compute_rmse(y_true: np.array, y_pred: np.array) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))
