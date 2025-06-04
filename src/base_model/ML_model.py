from src.data_loader.data_loader import load_train_data
import numpy as np

from sklearn.multioutput import MultiOutputRegressor

X_train, X_val, Y_val, sensor_cols = load_train_data('data/BATADAL/train_dataset.csv')

def threshold_select(model) :
    
    X_val_hat = model.predict(X_val)
    errors = np.mean((X_val - X_val_hat) ** 2, axis=1)
    benign_errors = errors[Y_val == 0]
    tau = np.percentile(benign_errors, 99.5)
    # print(f"Threshold τ (99.5th percentile): {tau:.6f}")

    return tau

'''LinearRegression - Ridge'''
from sklearn.linear_model import Ridge
def train_linear_regression_ridge_MOR(penalize_l2_norm = 1.0) :

    base_model = Ridge(penalize_l2_norm)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, X_train)

    return model
'''--------------------------'''

'''Suport Vector Regression - SVR'''
from sklearn.svm import SVR
def train_SVR(penalize=1.0, epsilon=0.1) :
    
    # init svr and multi svr model
    svr = SVR(kernel='rbf', C=penalize, epsilon=epsilon)    # needs tuning for best performance

    multi_svr = MultiOutputRegressor(svr, n_jobs=2)

    return multi_svr
'''--------------------------'''

'''Adaboost - Ridge'''
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

def train_adaBoost(max_depth, learning_rate) :

    base_regressor = DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=42
    )

    ada = AdaBoostRegressor(
        n_estimators=200,
        learning_rate=learning_rate,
        estimator=base_regressor,
        random_state=42
    )
    
    return MultiOutputRegressor(ada, n_jobs=4)

'''--------------------------'''

'''XGBoost - XGBRegressor'''
from xgboost import XGBRegressor

def train_XGBoost(max_depth=3, learning_rate=0.1, n_estimators=100):
    """
    Trains a MultiOutput XGBoost Regressor.
    
    Args:
        max_depth (int): Maximum tree depth.
        learning_rate (float): Boosting learning rate.
        n_estimators (int): Number of gradient boosted trees.
    
    Returns:
        MultiOutputRegressor: Wrapped XGBoost model.
    """

    xgb = XGBRegressor(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=4
    )
    return MultiOutputRegressor(xgb, n_jobs=2)

'''--------------------------'''

'''Random Forest - RandomForestRegressor'''
from sklearn.ensemble import RandomForestRegressor

def train_RandomForest(n_estimators=100, max_depth=None):
    """
    Trains a MultiOutput Random Forest Regressor.
    
    Args:
        n_estimators (int): Number of trees in the forest.
        max_depth (int): Maximum depth of each tree.
    
    Returns:
        MultiOutputRegressor: Wrapped Random Forest model.
    """
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features=1,
        random_state=42,
        n_jobs=4
    )
    return MultiOutputRegressor(rf)