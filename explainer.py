import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from typing import Dict, Tuple, List

from logger_config import get_logger

logger = get_logger(__name__)

def calculate_shap_values(model: xgb.XGBRegressor, X_test: pd.DataFrame) -> np.ndarray:
    """
    Calculates SHAP values for the given test set using the trained TreeExplainer.

    Args:
        model: Trained XGBoost model.
        X_test: The testing features array.

    Returns:
        Array of SHAP values corresponding to the test set features.
    """
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X_test)

def calculate_process_importance(shap_values: np.ndarray, feature_names: List[str]) -> List[Tuple[str, float]]:
    """
    Calculates process-level importance by aggregating feature importances.

    Args:
        shap_values: The SHAP values matrix.
        feature_names: List of feature names.

    Returns:
        A list of tuples (Process Name, Aggregated Importance), sorted descending.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    param_importance = dict(zip(feature_names, mean_abs_shap))

    process_importance: Dict[str, float] = {}
    for param in feature_names:
        if param.startswith("Process_"):
            parts = param.split("_")
            if len(parts) >= 2:
                process_name = f"{parts[0]}_{parts[1]}"
                process_importance[process_name] = process_importance.get(process_name, 0.0) + param_importance[param]
        elif param == "Machine_Routing":
            process_importance["Process_3"] = process_importance.get("Process_3", 0.0) + param_importance[param]

    return sorted(process_importance.items(), key=lambda x: x[1], reverse=True)

def recommend_action(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    routing_labels: pd.Series,
    top_process: str
) -> Tuple[str, str]:
    """
    Recommends an action by grouping raw SHAP values.

    Args:
        shap_values: SHAP value matrix.
        X_test: Test features.
        routing_labels: Original routing strings.
        top_process: Name of the top process.

    Returns:
        Tuple of (Recommended Action, Action to Avoid).
    """
    top_process_params = []
    for col in X_test.columns:
        if col.startswith(top_process):
            top_process_params.append(col)
        elif top_process == "Process_3" and col == "Machine_Routing":
            top_process_params.append(col)

    param_indices = [X_test.columns.get_loc(p) for p in top_process_params if p in X_test.columns]
    top_process_shap_sum = shap_values[:, param_indices].sum(axis=1)

    action_df = pd.DataFrame({
        'Machine_Routing': routing_labels.values,
        'SHAP_Sum': top_process_shap_sum
    })

    grouped_shap = action_df.groupby('Machine_Routing')['SHAP_Sum'].mean()
    
    logger.info("Mean raw SHAP values by potential actions:")
    for action, val in grouped_shap.items():
        logger.info(f"  {action}: {val:.4f}")

    recommended_action = grouped_shap.idxmax()
    avoid_action = grouped_shap.idxmin()

    logger.info(f"-> Recommended Improvement Action: {recommended_action} (Highest positive mean SHAP: {grouped_shap[recommended_action]:.4f})")
    logger.info(f"-> Action to Avoid: {avoid_action} (Lowest negative mean SHAP: {grouped_shap[avoid_action]:.4f})")

    return recommended_action, avoid_action
