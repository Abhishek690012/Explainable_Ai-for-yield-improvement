import pandas as pd
import numpy as np

def main():
    M = 1000
    np.random.seed(42)
    
    K = 5
    params_per_process = 10
    
    data = {}
    for i in range(1, K + 1):
        for j in range(1, params_per_process + 1):
            col_name = f"Process_{i}_Param_{j}"
            data[col_name] = np.random.normal(loc=50.0, scale=10.0, size=M)
            
    df = pd.DataFrame(data)
    
    machines = ["Machine_A", "Machine_B", "Machine_C"]
    df["Machine_Routing"] = np.random.choice(machines, size=M)
    
    base_yield = np.random.normal(loc=90.0, scale=3.0, size=M)
    yield_drop = np.where(df["Machine_Routing"] == "Machine_B", 15.0, 0.0)
    final_yield = base_yield - yield_drop
    df["yield_score"] = np.clip(final_yield, 0.0, 100.0)
    
    filename = "dummy_factory_data.csv"
    df.to_csv(filename, index=False)
    print(f"Successfully generated synthetic dataset with {M} rows and saved to '{filename}'.")

if __name__ == "__main__":
    main()
