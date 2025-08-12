# src/data_preprocessor.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import Config
from log import logger

logger = logger.get_logger(__name__)

class DataPreprocessor:
    def __init__(self, time_steps: int = Config.TIME_STEPS):
        self.scaler = MinMaxScaler()
        self.time_steps = time_steps

    def scale_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Scales the 'current_price' column between 0 and 1.
        """
        try:
            logger.info("Scaling current_price column.")
            scaled = self.scaler.fit_transform(df[['current_price']])
            return scaled
        except Exception as e:
            logger.error(f"Error in scaling data: {str(e)}")
            raise

    def create_sequences(self, data: np.ndarray) -> tuple:
        """
        Creates LSTM sequences of shape (samples, time_steps, 1)
        """
        logger.info(f"Creating sequences with {self.time_steps} time steps.")
        X, y = [], []
        for i in range(self.time_steps, len(data)):
            X.append(data[i - self.time_steps:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def train_test_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Splits sequences into training and testing sets.
        """
        test_size = int(len(X) * Config.TEST_SIZE)
        logger.info(f"Splitting data into train and test with test size {Config.TEST_SIZE}")
        return (
            X[:-test_size], y[:-test_size],
            X[-test_size:], y[-test_size:]
        )

    def inverse_scale(self, data: np.ndarray) -> np.ndarray:
        """
        Converts scaled data back to original scale.
        """
        return self.scaler.inverse_transform(data)
