"""
Find all parquet files in this folder with 'lcoe' in the filename. If there is a column
"cpa_mw", create a new columns "capacity_mw" that is equal to "cpa_mw". If there is no such column,
print the filename.
"""

from pathlib import Path

import pandas as pd

parquet_files = Path(".").glob("**/*lcoe*.parquet")

for parquet_file in parquet_files:
    print(parquet_file)
    df = pd.read_parquet(parquet_file)
    if "cpa_mw" in df.columns:
        df["capacity_mw"] = df["cpa_mw"]
    else:
        print(parquet_file, "has no cpa_mw column")
    df.to_parquet(parquet_file)


# now do the same for csv files
csv_files = Path(".").glob("**/*lcoe*.csv")
for csv_file in csv_files:
    print(csv_file)
    df = pd.read_csv(csv_file)
    if "cpa_mw" in df.columns:
        df["capacity_mw"] = df["cpa_mw"]
    else:
        print(csv_file, "has no cpa_mw column")
    df.to_csv(csv_file, index=False)
