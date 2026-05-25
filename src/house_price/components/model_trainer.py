import os
import sys

import numpy as np

from sklearn.linear_model import (
    LinearRegression
)

from sklearn.tree import (
    DecisionTreeRegressor
)

from sklearn.ensemble import (
    RandomForestRegressor
)

from sklearn.metrics import (
    r2_score
)

from house_price.exception import (
    CustomException
)

from house_price.logger import logging

from house_price.utils import (
    save_object,
    evaluate_models
)


class ModelTrainer:

    def __init__(self):

        self.trained_model_file_path = (
            os.path.join(
                "artifacts",
                "model.pkl"
            )
        )

    def initiate_model_trainer(
        self,
        train_array,
        test_array
    ):

        try:

            logging.info(
                "Splitting training and test input data"
            )

            X_train = (
                train_array[:, :-1]
            )

            y_train = (
                train_array[:, -1]
            )

            X_test = (
                test_array[:, :-1]
            )

            y_test = (
                test_array[:, -1]
            )

            models = {

                "Linear Regression":
                LinearRegression(),

                "Decision Tree":
                DecisionTreeRegressor(),

                "Random Forest":
                RandomForestRegressor()
            }

            model_report = (
                evaluate_models(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    models
                )
            )

            best_model_score = max(
                sorted(
                    model_report.values()
                )
            )

            best_model_name = list(
                model_report.keys()
            )[
                list(
                    model_report.values()
                ).index(
                    best_model_score
                )
            ]

            best_model = models[
                best_model_name
            ]

            logging.info(
                f"Best model found: "
                f"{best_model_name}"
            )

            save_object(
                file_path=(
                    self
                    .trained_model_file_path
                ),
                obj=best_model
            )

            predicted = (
                best_model.predict(
                    X_test
                )
            )

            r2_square = (
                r2_score(
                    y_test,
                    predicted
                )
            )

            return r2_square

        except Exception as e:

            raise CustomException(
                e,
                sys
            )
