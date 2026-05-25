import os
import json
import pandas as pd
from pathlib import Path

def aggregate_results(results_dir="results", output_file="/data/home/${USER:-user}/run/openunlearning/openunlearning/open-unlearning-main/grid_search/unlearning_report.csv"):
    all_data = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Directory {results_dir} not found.")
        return

    # Recursively find all .json files
    json_files = list(results_path.glob("**/*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            
            entry = {
                "task_name": json_file.stem,
                "path": str(json_file.relative_to(results_path))
            }
            
            # Flatten metrics
            
            for k, v in data.items():
                if isinstance(v, (int, float, str)):
                    entry[k] = v
                elif isinstance(v, dict): # Nested dict
                    for sub_k, sub_v in v.items():
                        entry[f"{k}_{sub_k}"] = sub_v
            
            all_data.append(entry)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    if not all_data:
        print("No data found to aggregate.")
        return

    df = pd.DataFrame(all_data)
    
    # Determine target columns
    if "task_name" in df.columns:
        df = df.sort_values("task_name")

    # Build rows
    df.to_csv(output_file, index=False)
    df.to_excel(output_file.replace(".csv", ".xlsx"), index=False)
    
    print(f"Aggregation complete. Saved to {output_file} and .xlsx")
    print("\nSummary (Top 5 rows):")
    print(df.head())

if __name__ == "__main__":
    
    aggregate_results(results_dir="/data/home/${USER:-user}/run/openunlearning/openunlearning/open-unlearning-main/grid_search")
