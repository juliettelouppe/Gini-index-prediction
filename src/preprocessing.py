from pathlib import Path
import pandas as pd

RAW_FILE = Path("data") / "wb_inequality_data.xlsx"
CLEAN_FILE = Path("data") / "inequality_clean.xlsx"


def clean_inequality_data(
    input_path: Path = RAW_FILE,
    output_path: Path = CLEAN_FILE,
    min_gini_per_country: int = 10,
    start_year: int = 1995,
    end_year: int = 2020,
) -> pd.DataFrame:

# Clean and preprocess the World Bank inequality dataset
    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"Loading raw data from: {input_path}")
    df = pd.read_excel(input_path)

    print("Initial shape:", df.shape)

# 1. Keep only countries with sufficiently Gini values  
    gini_counts = df[df["gini"].notna()].groupby("country")["gini"].count()
    valid_countries = gini_counts[gini_counts >= min_gini_per_country].index

    print("Number of countries with ≥", min_gini_per_country, "GINI values:", len(valid_countries))
    df = df[df["country"].isin(valid_countries)]
    print("After country filter:", df.shape)

# 2. Keep only observations within the chosen year window
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
    print(f"After year filter ({start_year}-{end_year}):", df.shape)

 # 3. Drop rows without a Gini value  
    df = df[df["gini"].notna()]
    print("After dropping rows without Gini:", df.shape)

#  Sort for correct interpolation  
    df = df.sort_values(["country", "year"])

# 4. Select numeric columns to interpolate 
    numeric_cols = df.select_dtypes(include="number").columns.tolist()


    for col in ["gini", "year"]:
        if col in numeric_cols:
            numeric_cols.remove(col)

    print("Numeric columns to interpolate:", numeric_cols)

# 5. Interpolation of missing values by country  
    for col in numeric_cols:
        df[col] = df.groupby("country")[col].transform(
            lambda s: s.interpolate(limit_direction="both")
        )

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    print("\n Number of NA per column after cleaning:")
    print(df.isna().sum())

# 6. Save cleaned dataset
    output_path.parent.mkdir(exist_ok=True)
    df.to_excel(output_path, index=False)

    print(f"\nCleaned data exported to: {output_path}")
    print("Final shape:", df.shape)

    return df


if __name__ == "__main__":
    clean_inequality_data()
