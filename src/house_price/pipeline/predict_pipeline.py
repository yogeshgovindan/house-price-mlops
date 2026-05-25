import os
import sys

import pandas as pd

from house_price.exception import (
    CustomException
)

from house_price.logger import logging

from house_price.utils import (
    load_object
)


class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, features):

        try:

            model_path = os.path.join(
                "artifacts",
                "model.pkl"
            )

            preprocessor_path = (
                os.path.join(
                    "artifacts",
                    "preprocessor.pkl"
                )
            )

            logging.info(
                "Loading model and preprocessor"
            )

            model = load_object(
                model_path
            )

            preprocessor = load_object(
                preprocessor_path
            )

            logging.info(
                "Transforming input features"
            )

            data_scaled = (
                preprocessor.transform(
                    features
                )
            )

            prediction = (
                model.predict(
                    data_scaled
                )
            )

            return prediction

        except Exception as e:

            raise CustomException(
                e,
                sys
            )


class CustomData:

    def __init__(
        self,
        longitude,
        latitude,
        housing_median_age,
        total_rooms,
        total_bedrooms,
        population,
        households,
        median_income,
        ocean_proximity
    ):

        self.longitude = longitude
        self.latitude = latitude
        self.housing_median_age = (
            housing_median_age
        )

        self.total_rooms = (
            total_rooms
        )

        self.total_bedrooms = (
            total_bedrooms
        )

        self.population = (
            population
        )

        self.households = (
            households
        )

        self.median_income = (
            median_income
        )

        self.ocean_proximity = (
            ocean_proximity
        )

    def get_data_as_dataframe(
        self
    ):

        try:

            custom_data_input_dict = {

                "longitude":
                [self.longitude],

                "latitude":
                [self.latitude],

                "housing_median_age":
                [
                    self
                    .housing_median_age
                ],

                "total_rooms":
                [
                    self.total_rooms
                ],

                "total_bedrooms":
                [
                    self.total_bedrooms
                ],

                "population":
                [
                    self.population
                ],

                "households":
                [
                    self.households
                ],

                "median_income":
                [
                    self.median_income
                ],

                "ocean_proximity":
                [
                    self
                    .ocean_proximity
                ]
            }

            return pd.DataFrame(
                custom_data_input_dict
            )

        except Exception as e:

            raise CustomException(
                e,
                sys
            )
