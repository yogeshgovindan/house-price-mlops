import os
import sys

import pandas as pd

from house_price.exception import CustomException
from house_price.logger import logging


class DataValidation:

    def __init__(self, train_path, test_path):

        self.train_path = train_path
        self.test_path = test_path

    def validate_dataset(self):

        try:

            logging.info(
                "Data validation started"
            )

            train_df = pd.read_csv(
                self.train_path
            )

            test_df = pd.read_csv(
                self.test_path
            )

            required_columns = [
                "longitude",
                "latitude",
                "housing_median_age",
                "total_rooms",
                "median_house_value"
            ]

            for column in required_columns:

                if column not in train_df.columns:

                    raise Exception(
                        f"Missing column: {column}"
                    )

            if train_df.empty:

                raise Exception(
                    "Training data is empty"
                )

            if test_df.empty:

                raise Exception(
                    "Testing data is empty"
                )

            logging.info(
                "Data validation completed"
            )

            return True

        except Exception as e:

            raise CustomException(
                e,
                sys
            )
