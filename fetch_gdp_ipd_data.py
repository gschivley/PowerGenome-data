#!/usr/bin/env python3
"""Fetch US GDP Implicit Price Deflator (GDP-IPD) data and create/update
data/dollar_year_adjustment.csv.

Uses FRED (Federal Reserve Bank of St. Louis) as the primary data source,
with the BEA NIPA API as a fallback. Stores annual GDP-IPD index values
(base year 2017 = 100) for each year, starting from 1980.

GDP-IPD is used by AEO and ATB for their dollar-year adjustments.

Usage
-----
    # Update with any new years not already in the CSV
    python fetch_gdp_ipd_data.py

    # Refetch everything from scratch
    python fetch_gdp_ipd_data.py --force

    # Write to a different path
    python fetch_gdp_ipd_data.py --output /path/to/dollar_year_adjustment.csv
"""

import argparse
import logging
import os
from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent / "data" / "dollar_year_adjustment.csv"

# FRED: GDP Implicit Price Deflator, Annual (Index 2017=100)
# Public CSV endpoint — no API key required.
FRED_SERIES_ID = "A191RD3A086NBEA"
FRED_CSV_URL = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={FRED_SERIES_ID}"

# BEA NIPA API — optional API key via BEA_API_KEY env var.
# Table 1.1.9: Implicit Price Deflators for Gross Domestic Product
BEA_API_URL = "https://apps.bea.gov/api/data"
BEA_TABLE_NAME = "T10109"
BEA_LINE_NUMBER = "1"  # Line 1 = Gross domestic product


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------


def _fetch_from_fred(start_year: int) -> pd.DataFrame:
    """Return a DataFrame of annual GDP-IPD values from FRED.

    FRED returns all available annual observations in a single request.
    Only years from *start_year* onward with a non-NaN value are included.
    The current (incomplete) year is excluded.
    """
    logger.debug("Requesting GDP-IPD data from FRED: %s", FRED_CSV_URL)
    resp = requests.get(FRED_CSV_URL, timeout=30)
    resp.raise_for_status()

    df = pd.read_csv(StringIO(resp.text))
    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = next((c for c in df.columns if "date" in c), df.columns[0])
    value_col = next((c for c in df.columns if c != date_col), df.columns[1])
    df = df.rename(columns={date_col: "date", value_col: "value"})
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["year"] = df["date"].dt.year

    current_year = date.today().year
    df = df[(df["year"] >= start_year) & (df["year"] < current_year)]
    df = df.dropna(subset=["value"])

    annual = df[["year", "value"]].sort_values("year").reset_index(drop=True)
    logger.debug("FRED returned %d annual GDP-IPD values", len(annual))
    return annual


def _fetch_from_bea(start_year: int, end_year: int) -> pd.DataFrame:
    """Return a DataFrame of annual GDP-IPD values from the BEA NIPA API.

    Requires a BEA API key available via the BEA_API_KEY environment variable.
    Raises RuntimeError if the key is not set.
    """
    api_key = os.environ.get("BEA_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "BEA_API_KEY environment variable is not set; cannot use BEA fallback."
        )

    params = {
        "UserID": api_key,
        "method": "GetData",
        "datasetname": "NIPA",
        "TableName": BEA_TABLE_NAME,
        "Frequency": "A",
        "Year": ",".join(str(y) for y in range(start_year, end_year + 1)),
        "ResultFormat": "json",
    }
    logger.debug("Requesting GDP-IPD data from BEA NIPA API")
    resp = requests.get(BEA_API_URL, params=params, timeout=60)
    resp.raise_for_status()

    payload = resp.json()
    if "BEAAPI" not in payload or "Results" not in payload["BEAAPI"]:
        raise RuntimeError(f"Unexpected BEA API response structure: {payload}")

    data = payload["BEAAPI"]["Results"].get("Data", [])
    records = []
    for row in data:
        if row.get("LineNumber") == BEA_LINE_NUMBER:
            try:
                year = int(row["TimePeriod"])
                value = float(row["DataValue"].replace(",", ""))
                records.append({"year": year, "value": value})
            except (KeyError, ValueError):
                continue

    if not records:
        return pd.DataFrame(columns=["year", "value"])

    annual = (
        pd.DataFrame(records)
        .drop_duplicates(subset=["year"])
        .sort_values("year")
        .reset_index(drop=True)
    )
    logger.debug("BEA returned %d annual GDP-IPD values", len(annual))
    return annual


# ---------------------------------------------------------------------------
# Main update logic
# ---------------------------------------------------------------------------


def update_gdp_ipd_data(
    force: bool = False, output_path: Path = DATA_PATH
) -> pd.DataFrame:
    """Fetch GDP-IPD data and write/update *output_path*.

    Parameters
    ----------
    force:
        When ``True``, refetch all data from 1980 regardless of what is
        already on disk.
    output_path:
        Destination CSV file.

    Returns
    -------
    pd.DataFrame
        The complete GDP-IPD dataset written to disk.
    """
    existing: pd.DataFrame | None = None
    start_year = 1980

    if output_path.exists() and not force:
        existing = pd.read_csv(output_path)
        existing_max = int(existing["year"].max())
        current_year = date.today().year
        if existing_max >= current_year - 1:
            logger.info(
                "GDP-IPD data already up to date through %d — nothing to fetch.",
                existing_max,
            )
            return existing
        start_year = existing_max + 1
        logger.info(
            "Existing data through %d; fetching from %d onward.",
            existing_max,
            start_year,
        )

    end_year = date.today().year - 1

    # Try FRED first (no rate limits, simpler API)
    new_data: pd.DataFrame | None = None
    fred_err: Exception | None = None
    try:
        new_data = _fetch_from_fred(start_year=start_year)
    except Exception as exc:
        fred_err = exc
        logger.warning(
            "FRED fetch failed (%s); falling back to BEA NIPA API.", exc
        )

    if new_data is None or new_data.empty:
        try:
            new_data = _fetch_from_bea(start_year=start_year, end_year=end_year)
        except Exception as bea_err:
            if new_data is None:
                raise RuntimeError(
                    "Both FRED and BEA fetches failed. "
                    f"FRED error: {fred_err}. BEA error: {bea_err}."
                ) from bea_err
            logger.warning("BEA fallback also failed: %s", bea_err)

    if existing is not None and not force:
        combined = (
            pd.concat([existing, new_data], ignore_index=True)
            .drop_duplicates(subset=["year"], keep="last")
            .sort_values("year")
            .reset_index(drop=True)
        )
    else:
        combined = new_data.sort_values("year").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False, float_format="%g")

    logger.info(
        "Saved %d years of GDP-IPD data (%d–%d) to %s",
        len(combined),
        int(combined["year"].min()),
        int(combined["year"].max()),
        output_path,
    )
    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch US GDP Implicit Price Deflator data and create/update "
            "data/dollar_year_adjustment.csv."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refetch all data from 1980, overwriting the existing file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_PATH,
        help=f"Output CSV path (default: {DATA_PATH})",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    df = update_gdp_ipd_data(force=args.force, output_path=args.output)

    # Print a summary of the newest rows
    print(df.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
