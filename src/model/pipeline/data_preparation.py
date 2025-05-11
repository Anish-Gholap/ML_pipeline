import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from loguru import logger

from model.pipeline.data_collection import load_data_from_db

def prepare_data():
    '''
    1. Add the income_cat column
    2. Stratify split
    3. Preprocess splits
    4. Return splits
    '''
    logger.info("Preparing dataset splits")

    #df = load_housing_data()
    df = load_data_from_db()

    df["income_cat"] = pd.cut(df["median_income"], bins=[0, 1.5, 3.0, 4.5, 6, np.inf], labels=[1, 2, 3, 4, 5])
    strat_train_set, strat_test_set = train_test_split(df, test_size=0.2, stratify=df["income_cat"], random_state=42)

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    X_train = strat_train_set.drop("median_house_value", axis = 1)
    y_train = strat_train_set["median_house_value"].copy()

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    return X_train, y_train, X_test, y_test