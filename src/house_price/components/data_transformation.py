import os
import sys

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from house_price.exception import (
    CustomException
)

from house_price.logger import logging
from house_price.utils import save_object


class DataTransformation:

    def __init__(self):

        self.preprocessor_obj_file_path = (
            os.path.join(
                "artifacts",
                "preprocessor.pkl"
            )
        )

    def get_data_transformer_object(self):

        try:

            logging.info(
                "Data transformation initiated"
            )

            numerical_columns = [
                "longitude",
                "latitude",
                "housing_median_age",
                "total_rooms",
                "total_bedrooms",
                "population",
                "households",
                "median_income"
            ]

            categorical_columns = [
                "ocean_proximity"
            ]

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "one_hot_encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    ),
                    (
                        "scaler",
                        StandardScaler(
                            with_mean=False
                        )
                    )
                ]
            )

            logging.info(
                "Pipelines created successfully"
            )

            preprocessor = (
                ColumnTransformer(
                    transformers=[
                        (
                            "num_pipeline",
                            num_pipeline,
                            numerical_columns
                        ),
                        (
                            "cat_pipeline",
                            cat_pipeline,
                            categorical_columns
                        )
                    ]
                )
            )

            return preprocessor

        except Exception as e:

            raise CustomException(
                e,
                sys
            )

    def initiate_data_transformation(
        self,
        train_path,
        test_path
    ):

        try:

            # Load train and test data
            train_df = pd.read_csv(
                train_path
            )

            test_df = pd.read_csv(
                test_path
            )

            logging.info(
                "Train and test data loaded successfully"
            )

            preprocessing_obj = (
                self.get_data_transformer_object()
            )

            target_column_name = (
                "median_house_value"
            )

            # Split input and target
            input_feature_train_df = (
                train_df.drop(
                    columns=[
                        target_column_name
                    ]
                )
            )

            target_feature_train_df = (
                train_df[
                    target_column_name
                ]
            )

            input_feature_test_df = (
                test_df.drop(
                    columns=[
                        target_column_name
                    ]
                )
            )

            target_feature_test_df = (
                test_df[
                    target_column_name
                ]
            )

            logging.info(
                "Applying preprocessing object"
            )

            # Fit transform train data
            input_feature_train_arr = (
                preprocessing_obj.fit_transform(
                    input_feature_train_df
                )
            )

            # Transform test data
            input_feature_test_arr = (
                preprocessing_obj.transform(
                    input_feature_test_df
                )
            )

            logging.info(
                "Preprocessing completed"
            )

            # Combine features + target
            train_arr = np.c_[
                input_feature_train_arr,
                np.array(
                    target_feature_train_df
                )
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.array(
                    target_feature_test_df
                )
            ]

            # Save preprocessor object
            save_object(
                file_path=(
                    self.preprocessor_obj_file_path
                ),
                obj=preprocessing_obj
            )

            logging.info(
                "Preprocessor saved successfully"
            )

            return (
                train_arr,
                test_arr,
                self.preprocessor_obj_file_path
            )

        except Exception as e:

            raise CustomException(
                e,
                sys
            )
