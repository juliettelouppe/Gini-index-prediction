from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from src.preprocessing import clean_inequality_data
from src.split_data import split_data
from src.models import compare_models


def main():

    print("=======================================================")
    print(" Gini Index Prediction: Model Comparison")
    print("=======================================================\n")

   
    print("1. Loading and preprocessing data...")
    df = clean_inequality_data()
    print(f"   Dataset shape: {df.shape}")

   
    print("\n2. Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df, "gini")
    print(f"   Train size: {X_train.shape}")
    print(f"   Test size:  {X_test.shape}")

   
    print("\n3. Training models...")
    results = compare_models(X_train, X_test, y_train, y_test)
    print("   ✓ All models trained")

    
    print("\n4. Evaluating models...")
    print(results)

    
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / "model_comparison.csv"
    if isinstance(results, pd.DataFrame):
        results.to_csv(output_path, index=False)
    else:
        pd.DataFrame(results).to_csv(output_path, index=False)

    print("\n=======================================================")
    try:
        best_model = results.sort_values("r2", ascending=False).iloc[0]
        print("\n=======================================================")
        print(f"Winner (selected by best R²): {best_model['model']}")
        print(f"R²:   {best_model['r2']:.3f}")
        print("=======================================================\n")
    except:
        print("Winner: Could not determine (results format unexpected)")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
