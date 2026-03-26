import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Tuple

from logger_config import get_logger

logger = get_logger(__name__)

def load_and_prep_data(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """
    Loads and preprocesses manufacturing data for model training.

    Args:
        filepath: The path to the CSV file containing the manufacturing data.

    Returns:
        A tuple containing X_train, X_test, y_train, y_test, df.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        logger.error(f"Data file not found at {filepath}")
        raise FileNotFoundError(f"Data file not found at {filepath}")

    if df.isnull().values.any():
        logger.warning("Missing values detected. Numeric fields will be imputed with their medians.")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        if df.select_dtypes(include=['object']).isnull().values.any():
            df.dropna(subset=df.select_dtypes(include=['object']).columns, inplace=True)

    process_cols = [col for col in df.columns if col.startswith("Process_")]
    for col in process_cols:
        df[col] = df[col].astype(float)

    if "Machine_Routing" in df.columns:
        df["Machine_Routing_Original"] = df["Machine_Routing"]
        le = LabelEncoder()
        df["Machine_Routing"] = le.fit_transform(df["Machine_Routing"].astype(str))

    target_col = "yield_score"
    features = [col for col in df.columns if col not in [target_col, "Machine_Routing_Original"]]
    
    X = df[features]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, df

if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, full_df = load_and_prep_data("dummy_factory_data.csv")
    logger.info("Data loaded successfully.")
    logger.info(f"X_train shape: {X_tr.shape}, X_test shape: {X_te.shape}")
    logger.info(f"y_train shape: {y_tr.shape}, y_test shape: {y_te.shape}")
    logger.info(f"Columns in dataframe: {full_df.columns.tolist()}")
