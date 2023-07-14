from pydantic import BaseModel

class PredictionRequest(BaseModel):
    input_data: int

# The second part of the prototype is a script that makes a prediction using the latest
# trained model.
# ○ A scheduling service that ensures the delivery of the forecast daily before 12:00.
# ○ The location of the input data and forecast period extent should be arguments
# ○ The output can be saved in a storage solution of your choice

class TrainRequest(BaseModel):

#     Deliver a model training service. Some requirements:
# ○ The model is retrained every week
# ○ It should be possible to specify the location of the input dataset
# ○ The time extent of the train set should be an argument to the script
# ○ Trained model is stored in a suitable file system