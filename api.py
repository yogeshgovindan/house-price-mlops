from fastapi import FastAPI
from pydantic import BaseModel

from house_price.pipeline.predict_pipeline import (
    PredictPipeline,
    CustomData
)

app = FastAPI()


class HouseData(BaseModel):

    longitude: float
    latitude: float
    housing_median_age: float

    total_rooms: float
    total_bedrooms: float

    population: float
    households: float

    median_income: float
    ocean_proximity: str


@app.get("/")
def home():

    return {
        "message":
        "House Price Prediction API Running"
    }


@app.post("/predict")
def predict(data: HouseData):

    custom_data = CustomData(

        longitude=data.longitude,
        latitude=data.latitude,

        housing_median_age=(
            data.housing_median_age
        ),

        total_rooms=(
            data.total_rooms
        ),

        total_bedrooms=(
            data.total_bedrooms
        ),

        population=data.population,
        households=data.households,

        median_income=(
            data.median_income
        ),

        ocean_proximity=(
            data.ocean_proximity
        )
    )

    pred_df = (
        custom_data
        .get_data_as_dataframe()
    )

    prediction_pipeline = (
        PredictPipeline()
    )

    result = (
        prediction_pipeline
        .predict(pred_df)
    )

    return {

        "predicted_house_price":
        float(result[0])
    }
