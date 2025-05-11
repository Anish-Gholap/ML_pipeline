import joblib
import pandas as pd
from pathlib import Path
from model import build_model, column_ratio, ratio_pipeline, make_preprocessing_pipeline, ClusterSimilarity, ratio_name
from config import settings
from loguru import logger

class ModelService:
    def __init__(self):
        self.model = None

    def load_model(self):
        logger.info(f"Checking if model exists at {settings.model_path}/{settings.model_name}")
        model_path = Path(f'{settings.model_path}/{settings.model_name}.pkl')

        if not model_path.exists():
            logger.warning(f"Model not found at {settings.model_path} --> Building model {settings.model_name}")
            print("Model not found, building model")
            build_model()

        logger.info(f"Model {settings.model_name} exists --> loading model")
        self.model = joblib.load(model_path)

    def predict(self, input_parameters):
        logger.info("Making prediction!")
        return self.model.predict(input_parameters)