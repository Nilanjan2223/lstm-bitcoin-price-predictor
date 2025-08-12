# src/predictor.py

import os
import numpy as np
from tensorflow.keras.models import load_model
from config import Config
from data_preprocessor import DataPreprocessor
from log import logger

logger = logger.get_logger(__name__)


class CryptoPricePredictor:
    def __init__(self, coin_name: str):
        self.coin_name = coin_name
        self.model_path = os.path.join(Config.MODEL_DIR, f"{coin_name}_lstm_model.h5")
        self.preprocessor = DataPreprocessor()
        self.model = None

        if not os.path.exists(self.model_path):
            logger.error(f"No trained model found for {coin_name} at {self.model_path}")
            raise FileNotFoundError(f"Model for {coin_name} not found.")

        self.model = load_model(self.model_path)
        logger.info(f"Loaded model for {coin_name}")

    def predict_next_price(self, df):
        """
        Takes the full historical DataFrame, scales it, creates the latest sequence,
        and predicts the next price.
        """
        try:
            logger.info("Starting prediction process...")

            scaled_data = self.preprocessor.scale_data(df)
            last_sequence = scaled_data[-Config.TIME_STEPS:]
            input_data = np.expand_dims(last_sequence, axis=0)  # Shape: (1, 60, 1)

            predicted_scaled = self.model.predict(input_data)
            predicted_price = self.preprocessor.inverse_scale(predicted_scaled)[0][0]

            logger.info(f"Predicted price for {self.coin_name}: {predicted_price:.2f}")
            return predicted_price

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
