# Streamlit dashboard app that allows users to select a coin and view predicted price
import streamlit as st
from data_fetcher import CryptoDataFetcher
from predictor import CryptoPricePredictor
#from log import logger
import sys
import os
#logger = logger.get_logger(__name__)

# Supported coins
COINS = [
    "Bitcoin", "Ethereum", "XRP", "Tether", "BNB",
    "Solana", "USDC", "Lido Staked Ether", "TRON", "Dogecoin"
]

def main():
    st.title("📈 Crypto Price Predictor")
    st.markdown("Select a cryptocurrency to view its predicted next price.")

    coin = st.selectbox("Select Coin", COINS)

    if st.button("Predict Price"):
        try:
            fetcher = CryptoDataFetcher()
            df = fetcher.fetch_coin_data(coin)
            fetcher.close_session()

            if df.empty:
                st.warning(f"No data found for {coin}")
                return

            predictor = CryptoPricePredictor(coin_name=coin)
            predicted_price = predictor.predict_next_price(df)

            st.success(f"Predicted next price for **{coin}** is: ₹{predicted_price:.2f}")

            st.line_chart(df.set_index("timestamp")["current_price"])

        except FileNotFoundError:
            st.error(f"Model for {coin} not found. Please train it first.")
        except Exception as e:
           # logger.error(f"Dashboard error: {str(e)}")
            st.error("Something went wrong. Check logs for details.")

if __name__ == "__main__":
    main()
"""

# Save dashboard script
dashboard_path = "crypto_price_predictor/dashboard/app.py"
with open(dashboard_path, "w") as f:
    f.write(dashboard_code)
"""
