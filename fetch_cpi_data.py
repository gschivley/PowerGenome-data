#!/usr/bin/env python3
"""Fetch US CPI-U data and create/update data/cpi_data.csv.

Uses FRED (Federal Reserve Bank of St. Louis) as the primary data source,
with the BLS public API as a fallback. Stores annual average CPI-U values
for each year, matching the format expected by PowerGenome.

Usage
-----
    # Update with any new years not already in the CSV
    python fetch_cpi_data.py

    # Refetch everything from scratch
    python fetch_cpi_data.py --force

    # Write to a different path
    python fetch_cpi_data.py --output /path/to/cpi_data.csv
"""

import argparse
import json
import logging
from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent / "data" / "cpi_data.csv"

# FRED: CPI for All Urban Consumers, All Items, Not Seasonally Adjusted
# Public CSV endpoint — no API key required.
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCNS"

# BLS public V1 API — no key required, but rate-limited.
BLS_SERIES_ID = "CUUR0000SA0"
BLS_API_URL = "https://api.bls.gov/publicAPI/v1/timeseries/data/"
BLS_MAX_YEARS_PER_REQUEST = 10


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------


def _fetch_from_fred(start_year: int) -> pd.DataFrame:
    """Return a DataFrame of annual-average CPI-U values from FRED.

    FRED returns all available monthly observations in a single request.
    Annual averages are computed as the mean of all 12 monthly values,
    matching the convention used by PowerGenome (period=12 denotes a
    12-month average, not December specifically).  Only years with all
    12 months of data are included.
    """
    logger.debug("Requesting CPI data from FRED: %s", FRED_CSV_URL)
    resp = requests.get(FRED_CSV_URL, timeout=30)
    resp.raise_for_status()

    df = pd.read_csv(StringIO(resp.text))
    # Normalise column names — FRED may return "DATE"/"date"/"observation_date"
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = next((c for c in df.columns if "date" in c), df.columns[0])
    value_col = next((c for c in df.columns if c != date_col), df.columns[1])
    df = df.rename(columns={date_col: "date", value_col: "value"})
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["year"] = df["date"].dt.year

    # Keep only years from start_year onward
    df = df[df["year"] >= start_year]

    # Annual average: include past years with ≥ 11 non-NaN months (one month
    # may occasionally be missing from FRED due to a pending BLS revision).
    current_year = date.today().year
    agg = df.groupby("year")["value"].agg(["mean", "count"])
    complete_years = agg[(agg["count"] >= 11) & (agg.index < current_year)].index
    if (
        len(agg[(agg["count"] < 11) & (agg["count"] > 0) & (agg.index < current_year)])
        > 0
    ):
        missing_counts = agg[
            (agg["count"] < 11) & (agg["count"] > 0) & (agg.index < current_year)
        ]
        logger.warning(
            "Skipping %d year(s) with fewer than 11 months of FRED data: %s",
            len(missing_counts),
            list(missing_counts.index),
        )

    annual = (
        agg.loc[complete_years, ["mean"]]
        .rename(columns={"mean": "value"})
        .reset_index()
    )
    annual["period"] = 12
    annual = (
        annual[["year", "period", "value"]].sort_values("year").reset_index(drop=True)
    )

    logger.debug("FRED returned %d complete annual averages", len(annual))
    return annual


def _fetch_from_bls(start_year: int, end_year: int) -> pd.DataFrame:
    """Return a DataFrame of annual-average CPI-U values from the BLS V1 API.

    The V1 API is limited to 10-year windows per request, so multiple
    requests are made when the range spans more than 10 years. Annual
    averages are computed as the mean of all 12 monthly values; only years
    with all 12 months present are included.
    """
    records: list[dict] = []
    year = start_year

    while year <= end_year:
        batch_end = min(year + BLS_MAX_YEARS_PER_REQUEST - 1, end_year)
        payload = json.dumps(
            {
                "seriesid": [BLS_SERIES_ID],
                "startyear": str(year),
                "endyear": str(batch_end),
            }
        )
        logger.debug("BLS request: %d–%d", year, batch_end)
        resp = requests.post(
            BLS_API_URL,
            data=payload,
            headers={"Content-type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()

        result = resp.json()
        if result.get("status") != "REQUEST_SUCCEEDED":
            messages = result.get("message", [])
            raise RuntimeError(f"BLS API error: {messages}")

        for series in result["Results"]["series"]:
            for item in series["data"]:
                # M01–M12 are monthly values; M13 is the annual average
                # (we compute our own average so we can verify completeness)
                period = item.get("period", "")
                if period.startswith("M") and period != "M13":
                    month = int(period[1:])
                    records.append(
                        {
                            "year": int(item["year"]),
                            "month": month,
                            "value": float(item["value"]),
                        }
                    )

        year = batch_end + 1

    if not records:
        return pd.DataFrame(columns=["year", "period", "value"])

    monthly = pd.DataFrame(records)
    counts = monthly.groupby("year")["month"].count()
    complete_years = counts[counts == 12].index
    annual = (
        monthly[monthly["year"].isin(complete_years)]
        .groupby("year")["value"]
        .mean()
        .reset_index()
    )
    annual["period"] = 12
    annual = (
        annual[["year", "period", "value"]].sort_values("year").reset_index(drop=True)
    )

    logger.debug("BLS returned %d complete annual averages", len(annual))
    return annual


# ---------------------------------------------------------------------------
# Main update logic
# ---------------------------------------------------------------------------


def update_cpi_data(force: bool = False, output_path: Path = DATA_PATH) -> pd.DataFrame:
    """Fetch CPI data and write/update *output_path*.

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
        The complete CPI dataset written to disk.
    """
    existing: pd.DataFrame | None = None
    start_year = 1980

    if output_path.exists() and not force:
        existing = pd.read_csv(output_path)
        existing_max = int(existing["year"].max())
        current_year = date.today().year
        if existing_max >= current_year - 1:
            # December of the previous year is the latest that can exist; if
            # we already have it, there is nothing to do (unless --force).
            logger.info(
                "CPI data already up to date through %d — nothing to fetch.",
                existing_max,
            )
            return existing
        start_year = existing_max + 1
        logger.info(
            "Existing data through %d; fetching from %d onward.",
            existing_max,
            start_year,
        )

    end_year = date.today().year

    # Try FRED first (no rate limits, simpler API)
    new_data: pd.DataFrame | None = None
    try:
        new_data = _fetch_from_fred(start_year=start_year)
    except Exception as fred_err:
        logger.warning("FRED fetch failed (%s); falling back to BLS API.", fred_err)

    if new_data is None or new_data.empty:
        try:
            new_data = _fetch_from_bls(start_year=start_year, end_year=end_year)
        except Exception as bls_err:
            if new_data is None:
                raise RuntimeError(
                    "Both FRED and BLS fetches failed. "
                    f"FRED error: {fred_err}. BLS error: {bls_err}."
                ) from bls_err
            # new_data is empty but not None — fall through with empty frame
            logger.warning("BLS fallback also failed: %s", bls_err)

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
        "Saved %d years of CPI data (%d–%d) to %s",
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
        description="Fetch US CPI-U data and create/update data/cpi_data.csv.",
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

    df = update_cpi_data(force=args.force, output_path=args.output)

    # Print a summary of the newest rows
    print(df.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
