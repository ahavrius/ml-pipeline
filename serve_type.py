from pydantic import BaseModel

class DataRequest(BaseModel):
    location_type: str = "local"
    location_path: str

class TimePeriod(BaseModel):
    start_datetime: str | None = None
    end_datetime: str | None = None

class PredictionRequest(BaseModel):
    input_data: DataRequest
    output_data: DataRequest
    time_period: TimePeriod
    saved_model: DataRequest

# The second part of the prototype is a script that makes a prediction using the latest
# trained model.
# ○ A scheduling service that ensures the delivery of the forecast daily before 12:00.
# ○ The location of the input data and forecast period extent should be arguments
# ○ The output can be saved in a storage solution of your choice

class TrainingRequest(BaseModel):
    input_data: DataRequest
    output_data: DataRequest
    time_period: TimePeriod

#     Deliver a model training service. Some requirements:
# ○ It should be possible to specify the location of the input dataset
# ○ The time extent of the train set should be an argument to the script
# ○ Trained model is stored in a suitable file system

### ○ The model is retrained every week


# model_config = {
#         "json_schema_extra": {
#             "examples": [
#                 {
#                     "name": "Foo",
#                     "description": "A very nice Item",
#                     "price": 35.4,
#                     "tax": 3.2,
#                 }
#             ]
#         }
#     }