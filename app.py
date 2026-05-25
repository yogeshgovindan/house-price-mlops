from house_price.pipeline.predict_pipeline import (
    PredictPipeline,
    CustomData
)


if __name__ == "__main__":

    data = CustomData(

        longitude=-122.23,
        latitude=37.88,
        housing_median_age=41,

        total_rooms=880,
        total_bedrooms=129,

        population=322,
        households=126,

        median_income=8.3252,

        ocean_proximity="NEAR BAY"
    )

    pred_df = (
        data.get_data_as_dataframe()
    )

    print(pred_df)

    predict_pipeline = (
        PredictPipeline()
    )

    result = (
        predict_pipeline.predict(
            pred_df
        )
    )

    print(
        "\nPredicted House Price:"
    )

    print(result[0])
