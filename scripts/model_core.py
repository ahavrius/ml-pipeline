import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from scripts.model_config import random_forest_init_config
from scripts.data_preparation import upload_data

    def save_predict(self):
        # parquit file
        pass

    def save_model(self, path):
        # create folder if needed
        # save parameters of api call
        # the data trained ...  + model parameters
        # save model itself with reproduceble name
        # save calculated matrics
        with open(f"{path}/model.pkl", 'wb') as file:
            pickle.dump(self.model, file)

    def load_model(self, path):
        with open(f"{path}/model.pkl", 'rb') as file:
            self.model = pickle.load(file)




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