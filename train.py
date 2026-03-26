import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

from logger_config import get_logger

logger = get_logger(__name__)

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> xgb.XGBRegressor:
    """
    Trains an XGBoost Regressor model and evaluates its performance.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_test: Testing features.
        y_test: Testing target.

    Returns:
        The trained XGBoost model.
    """
    model = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    logger.info(f"XGBoost Test MAE: {mae:.4f}")

    return model
