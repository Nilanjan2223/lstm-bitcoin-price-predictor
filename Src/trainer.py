# src/trainer.py
import mlflow
import mlflow.keras
import os
from tensorflow.keras.callbacks import ModelCheckpoint
from config import Config
from log import logger

class ModelTrainer:
    def __init__(self, model, coin_name: str):
        self.model = model
        self.coin_name = coin_name
        self.model_dir = Config.MODEL_DIR
        self.epochs = Config.EPOCHS
        self.batch_size = Config.BATCH_SIZE

    def train(self, X_train, y_train, X_val=None, y_val=None):
        model_path = os.path.join(self.model_dir, f"{self.coin_name}_lstm_model.h5")

        with mlflow.start_run(run_name=f"{self.coin_name}_lstm"):
            # Log parameters
            mlflow.log_param("epochs", self.epochs)
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("time_steps", Config.TIME_STEPS)
            mlflow.log_param("dropout_rate", 0.2)

            callbacks = [
                ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='loss', verbose=1)
            ]

            history = self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )

            # Log metrics
            for epoch in range(self.epochs):
                mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
                if 'val_loss' in history.history:
                    mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)

            # Log the model
            mlflow.keras.log_model(self.model, artifact_path="model")

        return history
