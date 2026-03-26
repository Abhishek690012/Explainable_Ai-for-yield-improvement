import warnings
from data_pipeline import load_and_prep_data
from train import train_xgboost
from explainer import calculate_shap_values, calculate_process_importance, recommend_action
from logger_config import get_logger

warnings.filterwarnings("ignore")

logger = get_logger(__name__)

def main():
    logger.info("=== Process Quality XAI Pipeline ===")
    
    logger.info("1. Loading and preprocessing data...")
    try:
        X_train, X_test, y_train, y_test, df = load_and_prep_data("dummy_factory_data.csv")
    except FileNotFoundError:
        return
        
    routing_labels_test = df.loc[X_test.index, "Machine_Routing_Original"]
    
    logger.info("2. Training XGBoost Model...")
    model = train_xgboost(X_train, y_train, X_test, y_test)
    
    logger.info("3. Calculating SHAP values...")
    shap_values = calculate_shap_values(model, X_test)
    
    logger.info("--- Step 1: Process Prioritization ---")
    ranked_processes = calculate_process_importance(shap_values, list(X_test.columns))
    for rank, (proc, imp) in enumerate(ranked_processes, 1):
        logger.info(f"{rank}. {proc}: {imp:.4f}")
        
    top_process = ranked_processes[0][0]
    logger.info(f"Top-ranked process identified: {top_process}")
    
    logger.info("--- Step 2: Action Selection ---")
    recommend_action(shap_values, X_test, routing_labels_test, top_process)

if __name__ == "__main__":
    main()
