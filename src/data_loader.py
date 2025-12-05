import pandas as pd
import requests
from pathlib import Path

# Dictionary mapping: World Bank indicator code
WB_INDICATORS = {
    "SI.POV.GINI": "gini",
    "NY.GDP.PCAP.CD": "gdp_pc",
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",
    "SP.POP.TOTL": "population",
    "SL.TLF.ACTI.FE.ZS": "female_labor_participation",
    "SL.TLF.CACT.ZS": "labor_participation",
    "SE.SEC.ENRR": "education_secondary_enrollment",
    "SP.URB.TOTL.IN.ZS": "urbanization",
    "NE.TRD.GNFS.ZS": "trade_openness",
    "GC.TAX.TOTL.GD.ZS": "tax_revenue",
}

def fetch_wb_indicator(indicator_code: str, col_name: str) -> pd.DataFrame:
# Downloads a single World Bank indicator using the API    
    print(f"  - downloading {col_name} ({indicator_code})")

 # Build API request URL
    url = (
        f"https://api.worldbank.org/v2/country/all/"
        f"indicator/{indicator_code}?format=json&per_page=20000"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()[1]

    rows = []
    for entry in data:
        year = entry["date"]
        if not year:
            continue
        try:
            year = int(year)
        except ValueError:
            continue

        value = entry["value"]
        country_name = entry["country"]["value"]
        country_code = entry["country"]["id"]

        rows.append(
            {
                "country": country_name,
                "country_code": country_code,
                "year": year,
                col_name: value,
            }
        )
 # Convert collected rows into a DataFrame
    df = pd.DataFrame(rows)
    return df

def build_wb_dataframe() -> pd.DataFrame:
    print("Downloading World Bank data (API)…")
    wb_df = None

 # Loop through all selected indicators
    for code, name in WB_INDICATORS.items():
        df = fetch_wb_indicator(code, name)
        if wb_df is None:
            wb_df = df
        else:
            wb_df = wb_df.merge(
                df, on=["country", "country_code", "year"], how="outer"
            )
# Keep only the years we want for the project
    wb_df = wb_df[(wb_df["year"] >= 1990) & (wb_df["year"] <= 2023)]
    return wb_df

def main():
# Download and assemble the complete dataset 
    wb_df = build_wb_dataframe()

    output_path = "data/wb_inequality_data.xlsx"   
    wb_df.to_excel(output_path, index=False)

    print(f"\nWorld Bank data exported to: {output_path}")
    print("Rows:", wb_df.shape[0])
    print("Columns:", wb_df.shape[1])

if __name__ == "__main__":
    main()
