# src/data_fetcher.py

import pandas as pd
from sqlalchemy import text
from db_connector import DatabaseConnector
from log import logger

logger = logger.get_logger(__name__)

class CryptoDataFetcher:
    def __init__(self):
        self.db = DatabaseConnector()
        self.session = self.db.get_session()

    def fetch_coin_data(self, coin_name: str) -> pd.DataFrame:
        """
        Fetches historical data for a given cryptocurrency name.
        Returns a pandas DataFrame with timestamp and current_price.
        """
        try:
            query = text("""
                SELECT timestamp, current_price
                FROM CryptoMarketData
                WHERE coin_id = :coin_name
                ORDER BY timestamp ASC
            """)

            logger.info(f"Fetching data for coin: {coin_name}")
            result = self.session.execute(query, {'coin_name': coin_name})
            df = pd.DataFrame(result.fetchall(), columns=["timestamp", "current_price"])

            if df.empty:
                logger.warning(f"No data found for {coin_name}")
            else:
                logger.info(f"Fetched {len(df)} rows for {coin_name}")
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            return df

        except Exception as e:
            logger.error(f"Error fetching data for {coin_name}: {str(e)}")
            raise

    def close_session(self):
        self.session.close()
        logger.debug("Database session closed.")
