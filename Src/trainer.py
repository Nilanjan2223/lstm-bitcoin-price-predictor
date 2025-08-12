# src/trainer.py

import os
from tensorflow.keras.callbacks import ModelCheckpoint
from config import Config
from log import logger

logger = logger.get_logger(__name__)

class ModelTrainer:
    def __init__(self, model, coin_name: str):
        self.model = model
        self.coin_name = coin_name
        self.model_dir = Config.MODEL_DIR
        self.epochs = Config.EPOCHS
        self.batch_size = Config.BATCH_SIZE

        os.makedirs(self.model_dir, exist_ok=True)
        logger.info(f"ModelTrainer initialized for {coin_name}")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the LSTM model and saves it to disk.
        """
        try:
            model_path = os.path.join(self.model_dir, f"{self.coin_name}_lstm_model.h5")
            logger.info(f"Model will be saved to: {model_path}")

            callbacks = [
                ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='loss', verbose=1)
            ]

            logger.info("Starting model training...")
            history = self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(X_val, y_val) if X_val is not None else None,
                callbacks=callbacks,
                verbose=1
            )

            logger.info("Model training completed successfully.")
            return history

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
