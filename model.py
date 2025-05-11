import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from data_preparation import prepare_data
from config import settings
from loguru import logger

def build_model():
    '''
    Trains the model and saves it

    1. Load the splits
    2. Preprocess the training data
    3. Evaluate model
    4. Train the model
    5. Save the model
    '''
    logger.info("Building Model")

    # 1. Load the splits
    X_train, y_train, X_test, y_test = prepare_data()

    # 2. Preprocess Training Data
    preprocessing = make_preprocessing_pipeline()

    # 3. Train the model
    model = train_model(X_train, y_train, preprocessing)

    # 4. Evaluate the model
    evaluate_model(X_test, y_test, model)

    # 5. Save the model
    save_model(model)




def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"] # feature names out

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio,
        feature_names_out=ratio_name),
        StandardScaler()
    )


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0,
        random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters,
        random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_,
            gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in
            range(self.n_clusters)]

def make_preprocessing_pipeline():
    logger.info('Preparing data processing pipeline')

    log_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler()
    )

    cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)

    default_num_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler()
    )

    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"))

    preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population", "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ], remainder=default_num_pipeline)  # one column remaining housing_median_age

    return preprocessing

def train_model(X_train, y_train, preprocessing):
    logger.info("Training Model")
    full_pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ('random_forest', RandomForestRegressor(random_state=42))
    ])
    params_distribs = {
        'preprocessing__geo__n_clusters': randint(low=3, high=50),
        'random_forest__max_features': randint(low=2, high=20)
    }

    logger.debug(f"params_distribs: {params_distribs}")

    random_search = RandomizedSearchCV(
        full_pipeline, params_distribs, n_iter=1, cv=3, scoring='neg_root_mean_squared_error', random_state=42
    )

    random_search.fit(X_train, y_train)

    model = random_search.best_estimator_ # includes preprocessing

    return model

def evaluate_model(X_test, y_test, model):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    logger.info(f"Evaluating Model; RMSE: {np.sqrt(mse)} ")
    print(f'RMSE: {np.sqrt(mse)}')

def save_model(model):
    logger.info(f"Saving model to {settings.model_path/settings.model_name}")
    joblib.dump(model, f'{settings.model_path/settings.model_name}.pkl')