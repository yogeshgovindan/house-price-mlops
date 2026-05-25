import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from house_price.exception import CustomException
from house_price.logger import logging


class DataIngestion:

    def __init__(self):

        self.train_data_path = os.path.join(
            "artifacts",
            "train.csv"
        )

        self.test_data_path = os.path.join(
            "artifacts",
            "test.csv"
        )

        self.raw_data_path = os.path.join(
            "artifacts",
            "raw.csv"
        )

    def initiate_data_ingestion(self):

        logging.info(
            "Entered data ingestion method"
        )

        try:

            url = (
                "https://raw.githubusercontent.com/"
                "ageron/handson-ml/master/"
                "datasets/housing/housing.csv"
            )

            df = pd.read_csv(url)

            logging.info(
                "Dataset loaded successfully"
            )

            os.makedirs(
                os.path.dirname(
                    self.train_data_path
                ),
                exist_ok=True
            )

            df.to_csv(
                self.raw_data_path,
                index=False
            )

            logging.info(
                "Train test split initiated"
            )

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            train_set.to_csv(
                self.train_data_path,
                index=False
            )

            test_set.to_csv(
                self.test_data_path,
                index=False
            )

            logging.info(
                "Data ingestion completed"
            )

            return (
                self.train_data_path,
                self.test_data_path
            )

        except Exception as e:
            raise CustomException(
                e,
                sys
            )
