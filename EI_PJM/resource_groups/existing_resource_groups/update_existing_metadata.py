"""
Find all files with "metadata" in the filename in this folder. If the file is a CSV, read it in,
change the column "ipm_region" to "region", and the column "mw" to "capacity_mw". Then save the file
back out. If the file is not a CSV, print the filename.

"""

from pathlib import Path

import pandas as pd

metadata_files = Path(".").glob("**/*metadata*")
for metadata_file in metadata_files:
    if metadata_file.suffix == ".csv":
        df = pd.read_csv(metadata_file)
        df = df.rename(columns={"ipm_region": "region", "mw": "capacity_mw"})
        df.to_csv(metadata_file, index=False)
    else:
        print(metadata_file)
