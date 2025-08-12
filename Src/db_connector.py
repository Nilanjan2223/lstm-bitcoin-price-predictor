# src/db_connector.py
import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from config import Config
from log import logger
logger = logger.get_logger(__name__)

class DatabaseConnector:
    def __init__(self):
        try:
            logger.info("Initializing database engine.")
            self.engine = create_engine(Config.DB_URL)
            self.Session = sessionmaker(bind=self.engine)
            logger.info("Database engine initialized successfully.")
        except SQLAlchemyError as e:
            logger.error(f"Failed to initialize database engine: {str(e)}")
            raise

    def get_session(self):
        try:
            logger.debug("Creating new database session.")
            return self.Session()
        except SQLAlchemyError as e:
            logger.error(f"Failed to create session: {str(e)}")
            raise
