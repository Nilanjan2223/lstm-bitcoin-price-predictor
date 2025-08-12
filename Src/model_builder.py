# src/model_builder.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from log import logger

logger = logger.get_logger(__name__)

class LSTMModelBuilder:
    def __init__(self, input_shape, dropout_rate=0.2, lr=0.001):
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.learning_rate = lr
        logger.info(f"Initializing LSTMModelBuilder with input shape: {input_shape}")

    def build_model(self):
        """
        Builds and compiles the LSTM model.
        """
        try:
            logger.info("Building LSTM model.")
            model = Sequential()
            model.add(LSTM(64, return_sequences=True, input_shape=self.input_shape))
            model.add(Dropout(self.dropout_rate))
            model.add(LSTM(64))
            model.add(Dropout(self.dropout_rate))
            model.add(Dense(1))

            model.compile(
                loss='mean_squared_error',
                optimizer=Adam(learning_rate=self.learning_rate)
            )

            logger.info("LSTM model compiled successfully.")
            return model

        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise
