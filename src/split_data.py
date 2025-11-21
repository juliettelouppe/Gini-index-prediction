from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Split dataframe into X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test


def main():
    # 1) Charger les données nettoyées
    data_path = Path("data") / "inequality_clean.xlsx"   # adapte le nom si besoin
    print(f"Loading cleaned data from: {data_path}")
    df = pd.read_excel(data_path)

    print("Cleaned data shape:", df.shape)

    # 2) Split train / test (80/20)
    target_col = "gini"   # ⚠️ change le nom ici si ta colonne cible a un autre nom
    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column=target_col,
        test_size=0.2,
        random_state=42,
    )

    print("\nSplits shapes:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    # 3) Sauvegarder les splits dans results/
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    X_train.to_csv(results_dir / "X_train.csv", index=False)
    X_test.to_csv(results_dir / "X_test.csv", index=False)
    y_train.to_csv(results_dir / "y_train.csv", index=False)
    y_test.to_csv(results_dir / "y_test.csv", index=False)

    print(f"\nSplits saved in '{results_dir}/':")
    print(" - X_train.csv")
    print(" - X_test.csv")
    print(" - y_train.csv")
    print(" - y_test.csv")


if __name__ == "__main__":
    main()
