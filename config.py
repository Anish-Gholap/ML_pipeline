from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import DirectoryPath
from loguru import logger
from sqlalchemy import create_engine

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

    model_path: DirectoryPath
    model_name: str
    log_level: str
    db_conn_string: str
    housing_tablename: str

# Variable containing all our paths
settings = Settings()

logger.add("app.log", rotation="1 day", retention="2 days", compression="zip", level=settings.log_level)

engine = create_engine(settings.db_conn_string)