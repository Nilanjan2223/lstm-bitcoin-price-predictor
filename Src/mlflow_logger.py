# mlflow_logger.py
import os
import mlflow
from config import Config

# Set MLflow to log locally in the mlruns folder
mlflow.set_tracking_uri("file:///D:/PythonProject/mlruns")

def log_existing_model_as_artifact(coin_name: str, epochs=None, batch_size=None, time_steps=None, dropout_rate=None):
    """
    Logs an already-trained model (.h5) as an artifact in MLflow
    without using the model registry (local file logging only).
    """
    model_path = os.path.join(Config.MODEL_DIR, f"{coin_name}_lstm_model.h5")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found at {model_path}")

    with mlflow.start_run(run_name=f"{coin_name}_lstm_existing"):
        # Log key parameters
        mlflow.log_param("coin_name", coin_name)
        if epochs: mlflow.log_param("epochs", epochs)
        if batch_size: mlflow.log_param("batch_size", batch_size)
        if time_steps: mlflow.log_param("time_steps", time_steps)
        if dropout_rate: mlflow.log_param("dropout_rate", dropout_rate)

        # Log the existing model file as an artifact
        mlflow.log_artifact(model_path)

    print(f"✅ Existing model for {coin_name} logged to MLflow as an artifact.")

if __name__ == "__main__":
    # Example usage — change coin_name if needed
    log_existing_model_as_artifact(
        coin_name="Bitcoin",
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        time_steps=Config.TIME_STEPS,
        dropout_rate=0.2
    )

