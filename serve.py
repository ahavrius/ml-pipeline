from fastapi import FastAPI

from serve_type import PredictionRequest, TrainingRequest
from model_core import Model

app = FastAPI()

# add logging

@app.get("/forecast/")
def forecast(prediction_request: PredictionRequest):
    model = Model(prediction_request)
    model.predict_main()
    
    return {"message": "Model predicted successfully"}


@app.post("/train")
def train(training_request: TrainingRequest):
    model = Model(training_request)
    model.predict_main()
    
    return {"message": "Model trained successfully"}