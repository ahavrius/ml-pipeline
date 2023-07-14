import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from model_config import random_forest_init_config
from data_preparation import upload_data

# Mock ML model class
class Model:
    def __init__(self, request):
        self.model = None
        self.request = request
        self.features = None
        self.target = None
    
    def prepare_data(self):
        df = upload_data(self.request)
        
        self.features = generate_features(df)
        self.target = generate_target(df)

    def _train_step(self):
        self.model = get_power_forecaster()
        self.model.fit(self.features, self.target)
    
    def _predict_step(self):
        # if not self.model:
            # raise
        self.predict = self.model.predict(self.features)
    
    def calculate_metrics(self):
       rmse = compute_rmse(self.target, self.predict)

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

    def predict_main(self):
        self.prepare_data()
        self.load_model()
        self._predict_step()
        self.save_predict()

    def train_main(self):
        self.prepare_data()
        self._train_step()
        self._predict_step()
        self.calculate_metrics()
        self.save_model()





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