import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def main():
    # 1. Generate Dummy Manufacturing Data
    print("Generating dummy manufacturing data...")
    np.random.seed(42)
    M = 1000
    K = 5

    data_dict = {}
    process_features = {f"Process_{k}": [] for k in range(1, K+1)}

    # Generate continuous parameters
    for k in range(1, K+1):
        for n in range(1, 11):
            feature_name = f"Process_{k}_Param_{n}"
            data_dict[feature_name] = np.random.normal(loc=0, scale=1, size=M)
            process_features[f"Process_{k}"].append(feature_name)

    # Add categorical Machine_Routing for Process 3
    machines = ["Machine_A", "Machine_B", "Machine_C"]
    data_dict["Machine_Routing"] = np.random.choice(machines, size=M)
    process_features["Process_3"].append("Machine_Routing")

    df = pd.DataFrame(data_dict)

    # Generate yield score
    # Base yield is normally distributed around 90
    yield_score = np.random.normal(loc=90, scale=1.5, size=M)
    
    # Inject hidden, nonlinear rule: Machine B in Process 3 severely drops yield
    # The rest of the parameters have minimal impact (handled by the random noise)
    yield_score = np.where(df["Machine_Routing"] == "Machine_B", yield_score - 15, yield_score)
    df["yield_score"] = yield_score
    df["yield_score"] = np.clip(df["yield_score"], 0, 100)

    # Encode Machine_Routing properly for XGBoost
    # We will use label encoding so XGBoost can process it
    df["Machine_Routing_encoded"] = df["Machine_Routing"].map({"Machine_A": 0, "Machine_B": 1, "Machine_C": 2})

    # Prepare features for modeling
    feature_cols = [col for col in df.columns if col not in ["yield_score", "Machine_Routing"]]
    X = df[feature_cols]
    y = df["yield_score"]

    # 2. Train the Metamodel
    print("Training the metamodel...")
    # Keep the original Machine_Routing for grouping in Step 2
    X_train, X_test, y_train, y_test, routing_train, routing_test = train_test_split(
        X, y, df["Machine_Routing"], test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(random_state=42, n_estimators=100, max_depth=3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print("\n--- Metamodel Evaluation ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

    # 3. Calculate SHAP Values
    print("\nCalculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # 4. Step 1: Prioritize Processes (Process Importance)
    print("\n--- Step 1: Process Prioritization ---")
    # Mean absolute SHAP value for each parameter across the test set
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    param_importance = dict(zip(X_test.columns, mean_abs_shap))

    process_importance = {}
    for process, params in process_features.items():
        # Match parameter names with the encoded column used in the model
        model_params = [p + "_encoded" if p == "Machine_Routing" else p for p in params]
        total_importance = sum(param_importance[p] for p in model_params if p in param_importance)
        process_importance[process] = total_importance

    # Print ranked list of processes
    ranked_processes = sorted(process_importance.items(), key=lambda x: x[1], reverse=True)
    for rank, (proc, imp) in enumerate(ranked_processes, 1):
        print(f"{rank}. {proc}: {imp:.4f}")

    # 5. Step 2: Select Improvement Actions
    print("\n--- Step 2: Action Selection ---")
    top_process = ranked_processes[0][0]
    print(f"Top-ranked process: {top_process}")

    # For the top-ranked process, calculate the raw SHAP contribution
    top_process_params = process_features[top_process]
    model_params = [p + "_encoded" if p == "Machine_Routing" else p for p in top_process_params]
    
    # Extract indices for the top process parameters
    param_indices = [X_test.columns.get_loc(p) for p in model_params if p in X_test.columns]
    
    # Sum raw SHAP values of the top process parameters for each observation
    top_process_shap_sum = shap_values[:, param_indices].sum(axis=1)

    # Group test set observations by their Machine_Routing action
    action_df = pd.DataFrame({
        'Machine_Routing': routing_test.values,
        'SHAP_Sum': top_process_shap_sum
    })

    # Mean raw SHAP value per machine group
    grouped_shap = action_df.groupby('Machine_Routing')['SHAP_Sum'].mean()
    print("\nMean raw SHAP values of Process Parameters by action:")
    for action, val in grouped_shap.items():
        print(f"  {action}: {val:.4f}")

    recommended_action = grouped_shap.idxmax()
    avoid_action = grouped_shap.idxmin()

    print(f"\n-> Recommended Improvement Action: {recommended_action} (Highest positive mean SHAP: {grouped_shap[recommended_action]:.4f})")
    print(f"-> Action to Avoid: {avoid_action} (Lowest negative mean SHAP: {grouped_shap[avoid_action]:.4f})")

if __name__ == "__main__":
    main()
