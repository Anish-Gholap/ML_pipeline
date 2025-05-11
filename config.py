from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import FilePath, DirectoryPath, HttpUrl
from pathlib import Path
from loguru import logger

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

    dataset_url: str
    dataset_path: Path
    tarball_name: str
    dataset_csv_path: str
    model_path: DirectoryPath
    model_name: str
    log_level: str

# Variable containing all our paths
settings = Settings()

logger.add("app.log", rotation="1 day", retention="2 days", compression="zip", level=settings.log_level)