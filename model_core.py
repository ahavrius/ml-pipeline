import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from model_config import random_forest_init_config

# Mock ML model class
class MLModel:
    def __init__(self):
        # Initialize or load the ML model
        pass
    
    def predict(self, input_data):
        # Make predictions using the ML model
        return {"prediction": input_data * 2}


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

# save parameters of api call
# the data trained ...  + model parameters
# save model itself with reproduceble name
# save calculated matrics 
def save_model():
    pass

def generate_target(df: pd.DataFrame) -> pd.Series:
    return df["power"]

def get_power_forecaster() -> RandomForestRegressor:
    return RandomForestRegressor(**random_forest_init_config)

def compute_rmse(y_true: np.array, y_pred: np.array) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))