import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")


def split_data(df, target_column, test_size=0.2, random_state=42):


    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols.remove(target_column)

    X = df[numeric_cols]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def save_splits(X_train, X_test, y_train, y_test):
 

    RESULTS_DIR.mkdir(exist_ok=True)

    X_train.to_csv(RESULTS_DIR / "X_train.csv", index=False)
    X_test.to_csv(RESULTS_DIR / "X_test.csv", index=False)
    y_train.to_csv(RESULTS_DIR / "y_train.csv", index=False)
    y_test.to_csv(RESULTS_DIR / "y_test.csv", index=False)

    print("\nSplits saved in 'results/':")
    print(" - X_train.csv")
    print(" - X_test.csv")
    print(" - y_train.csv")
    print(" - y_test.csv")


if __name__ == "__main__":
    from src.preprocessing import clean_inequality_data

    df = clean_inequality_data()

    print("\nCleaning finished.")
    print("Cleaned data shape:", df.shape)

    X_train, X_test, y_train, y_test = split_data(df, "gini")

    print("\nSplits shapes:")
    print(" X_train:", X_train.shape)
    print(" X_test :", X_test.shape)
    print(" y_train:", y_train.shape)
    print(" y_test :", y_test.shape)

    save_splits(X_train, X_test, y_train, y_test)

# load with python -m src.split_data (module package src)