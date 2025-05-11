import pandas as pd
from model_service import ModelService
from loguru import logger

@logger.catch
def main():
    logger.info('Running application')
    ml_svc = ModelService()
    ml_svc.load_model()
    data = {
        "total_bedrooms": [1400],
        "total_rooms": [5000],
        "households": [600],
        "population": [2200],
        "median_income": [4.2],
        "latitude": [34.05],
        "longitude": [-118.25],
        "ocean_proximity": ["NEAR BAY"],
        "housing_median_age": [19]
    }

    X = pd.DataFrame(data)
    pred = ml_svc.predict(X)

    logger.info(f"Prediction: {pred}")

if __name__ == "__main__":
    main()
