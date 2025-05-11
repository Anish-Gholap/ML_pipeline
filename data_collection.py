import pandas as pd
import urllib.request
import tarfile
from pathlib import Path
from config import settings
from loguru import logger

def load_housing_data():
    logger.info(f"Checking for tarball file at {settings.dataset_path/settings.tarball_name}")
    tarball_path = Path(settings.dataset_path/settings.tarball_name)
    if not tarball_path.is_file():
        logger.info(f"Downloading dataset from {settings.dataset_url} to {settings.dataset_path}")
        Path(settings.dataset_path).mkdir(parents=True, exist_ok=True)
        url = settings.dataset_url
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path=settings.dataset_path)

    logger.info(f"Reading CSV from {settings.dataset_path/settings.dataset_csv_path}")
    return pd.read_csv(Path(settings.dataset_path/settings.dataset_csv_path))