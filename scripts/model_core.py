import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from mlflow.models import infer_signature
import mlflow

from scripts.model_config import random_forest_init_config


# import pickle
        # with open(f"{path}/model.pkl", 'wb') as file:
        #     pickle.dump(self.model, file)

    # def load_model(self, path):
    #     with open(f"{path}/model.pkl", 'rb') as file:
    #         self.model = pickle.load(file)


def train_model(features: pd.DataFrame, target: pd.Series, artifact_path: str):
    with mlflow.start_run() as run:
        # model creation
        model = get_power_forecaster()
        model.fit(X=features, y=target)
        predict = model.predict(features)
        rmse = compute_rmse(target.to_numpy(), predict)

        # model logging
        signature = infer_signature(features, predict)
        mlflow.log_params(random_forest_init_config)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, artifact_path=artifact_path, signature=signature)
    return run

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[["cloudcover_low", "cloudcover_mid", "cloudcover_high"]]
        .assign(
            hour=lambda df: df.index.hour.astype("float"),
            month=lambda df: df.index.month.astype("float"),
            minutes=lambda df: 60.0 * df.index.hour + df.index.minute,
            day=lambda df: df.index.dayofyear.astype("float"),
        )
    )

def generate_target(df: pd.DataFrame) -> pd.Series | None:
    column_name = "power"
    if column_name in df.columns:
        return df[column_name]
    return None

def get_power_forecaster() -> RandomForestRegressor:
    return RandomForestRegressor(**random_forest_init_config)

def compute_rmse(y_true: np.array, y_pred: np.array) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))