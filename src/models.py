# run with python -m src.models2 (package src)

from pathlib import Path
import os

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor

# Directory 
RESULTS_DIR = Path("results")


def _evaluate_regression(y_true, y_pred):

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae


# Linear regression model
def run_linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics train / test
    r2_train, rmse_train, mae_train = _evaluate_regression(y_train, y_train_pred)
    r2_test, rmse_test, mae_test = _evaluate_regression(y_test, y_test_pred)

    print("\n==================== Linear Regression ====================")
    print(f"TRAIN - R² = {r2_train:.4f}, RMSE = {rmse_train:.4f}, MAE = {mae_train:.4f}")
    print(f"TEST  - R² = {r2_test:.4f}, RMSE = {rmse_test:.4f}, MAE = {mae_test:.4f}")

    # Return metrics in a dictionary 
    return {
        "model": "Linear Regression",
        "r2_train": r2_train,
        "rmse_train": rmse_train,
        "mae_train": mae_train,
        "r2": r2_test,
        "rmse": rmse_test,
        "mae": mae_test,
    }


# Random Forest model
def run_random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics train / test
    r2_train, rmse_train, mae_train = _evaluate_regression(y_train, y_train_pred)
    r2_test, rmse_test, mae_test = _evaluate_regression(y_test, y_test_pred)

    print("\n==================== Random Forest ====================")
    print(f"TRAIN - R² = {r2_train:.4f}, RMSE = {rmse_train:.4f}, MAE = {mae_train:.4f}")
    print(f"TEST  - R² = {r2_test:.4f}, RMSE = {rmse_test:.4f}, MAE = {mae_test:.4f}")

    # Feature importances 
    importances = model.feature_importances_
    feature_names = list(X_train.columns)
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    rf_path = RESULTS_DIR / "rf_feature_importances.csv"
    importance_df.to_csv(rf_path, index=False)
    print(f"\nVariable importances RF saved in: {rf_path}")

    return {
        "model": "Random Forest",
        "r2_train": r2_train,
        "rmse_train": rmse_train,
        "mae_train": mae_train,
        "r2": r2_test,
        "rmse": rmse_test,
        "mae": mae_test,
    }


# XGBoost model
def run_xgboost(X_train, X_test, y_train, y_test):
    model = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics train / test
    r2_train, rmse_train, mae_train = _evaluate_regression(y_train, y_train_pred)
    r2_test, rmse_test, mae_test = _evaluate_regression(y_test, y_test_pred)

    print("\n==================== XGBoost ====================")
    print(f"TRAIN - R² = {r2_train:.4f}, RMSE = {rmse_train:.4f}, MAE = {mae_train:.4f}")
    print(f"TEST  - R² = {r2_test:.4f}, RMSE = {rmse_test:.4f}, MAE = {mae_test:.4f}")

    # Feature importances
    importances = model.feature_importances_
    feature_names = list(X_train.columns)
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    xgb_imp_path = RESULTS_DIR / "xgb_feature_importances.csv"
    importance_df.to_csv(xgb_imp_path, index=False)
    print(f"\nVariable importances XGBoost saved in: {xgb_imp_path}")

    return {
        "model": "XGBoost",
        "r2_train": r2_train,
        "rmse_train": rmse_train,
        "mae_train": mae_train,
        "r2": r2_test,
        "rmse": rmse_test,
        "mae": mae_test,
    }


# Models Comparison
def compare_models(X_train, X_test, y_train, y_test):
    RESULTS_DIR.mkdir(exist_ok=True)

    res_lin = run_linear_regression(X_train, X_test, y_train, y_test)
    res_rf = run_random_forest(X_train, X_test, y_train, y_test)
    res_xgb = run_xgboost(X_train, X_test, y_train, y_test)

    df_results = pd.DataFrame([res_lin, res_rf, res_xgb])

    # Round metrics
    df_results = df_results.round(4)

    # Filter by performance 
    if "r2" in df_results.columns:
        df_results = df_results.sort_values("r2", ascending=False).reset_index(drop=True)

    print("\n==================== Comparative table ====================\n")
    print(df_results)

    out_path = RESULTS_DIR / "model_comparison.csv"
    df_results.to_csv(out_path, index=False)
    print(f"\nComparison table saved in: {out_path}")

    return df_results


if __name__ == "__main__":
    from src.preprocessing import clean_inequality_data
    from src.split_data import split_data

    df = clean_inequality_data()
    X_train, X_test, y_train, y_test = split_data(df, "gini")
    compare_models(X_train, X_test, y_train, y_test)

# run with python -m src.models2 (package src)
