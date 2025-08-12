# Create a pipeline script that connects all parts together for testing

from data_fetcher import CryptoDataFetcher
from data_preprocessor import DataPreprocessor
from model_builder import LSTMModelBuilder
from trainer import ModelTrainer
from predictor import CryptoPricePredictor
from log import logger

logger = logger.get_logger(__name__)

def run_pipeline(coin_name: str):
    try:
        # Step 1: Fetch Dashboard
        fetcher = CryptoDataFetcher()
        df = fetcher.fetch_coin_data(coin_name)
        if df.empty:
            logger.warning(f"No data found for {coin_name}. Pipeline aborted.")
            return
        fetcher.close_session()

        # Step 2: Preprocess Dashboard
        preprocessor = DataPreprocessor()
        scaled_data = preprocessor.scale_data(df)
        X, y = preprocessor.create_sequences(scaled_data)
        X_train, y_train, X_test, y_test = preprocessor.train_test_split(X, y)

        # Step 3: Build Model
        input_shape = (X_train.shape[1], X_train.shape[2])
        builder = LSTMModelBuilder(input_shape=input_shape)
        model = builder.build_model()

        # Step 4: Train Model
        trainer = ModelTrainer(model=model, coin_name=coin_name)
        trainer.train(X_train, y_train, X_val=X_test, y_val=y_test)

        # Step 5: Predict Latest Price
        predictor = CryptoPricePredictor(coin_name=coin_name)
        predicted_price = predictor.predict_next_price(df)
        print(f"✅ Predicted Next Price for {coin_name}: ₹{predicted_price:.2f}")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")

if __name__ == "__main__":
    run_pipeline("Bitcoin")  # Change coin name as needed
"""

# Save the pipeline code to a file
pipeline_path = "crypto_price_predictor/src/pipeline.py"
with open(pipeline_path, "w") as f:
    f.write(pipeline_code)
"""

