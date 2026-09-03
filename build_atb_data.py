#!/usr/bin/env python3
"""Build NREL ATB technology cost and heat-rate reference data (2023-2025).

This script rebuilds the committed ATB outputs:

* ``data/technology_costs_atb.parquet``
* ``data/technology_heat_rates_nrelatb.csv``

For every requested ATB year it downloads (and caches under
``cache_data/atb/``) the official OEDI tidy parquet and the ATB Excel
workbook, then reproduces the conventions of the historical build scripts:

* Tidy summary records (WACC Real, CAPEX, Fixed O&M, Variable O&M, Heat
  Rate and, from 2024 on, OCC/GCC/CFC) are normalized to the committed
  PowerGenome schema with ``dollar_year = atb_year - 2`` and $/kW metrics
  converted to $/MW.
* From ATB 2024 on, CAPEX includes grid connection costs (GCC). Following
  the original scripts a construction finance factor
  ``CFF = CAPEX / (OCC + GCC)`` is stored alongside the original CAPEX as
  ``ATB_CAPEX`` and ``capex_mw`` is recalculated without GCC
  (``CAPEX - GCC * CFF``).
* Utility-scale battery power ($/kW) and energy ($/kWh) capital costs are
  extracted from the workbook because the tidy data does not separate
  them. Battery fixed O&M is 2.5% of the corresponding capex and variable
  O&M is zero, matching the committed battery rows.
* ATB 2025 split the financial case into R&D / R&D + TC / Exp / Exp + TC;
  only the R&D case is kept.

Fallback: when a historical year (2023 or 2024) can no longer be rebuilt
from the official sources - the download fails, the workbook layout
changed, or the hosted source was revised and no longer contains records
present in the committed data - the committed rows for that year are
reused and a warning names the reused artifact. ATB 2025 is required and
never falls back.

Run from the repo root:

    uv run python build_atb_data.py [--years 2023 2024 2025] [--verbose]
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Final
from urllib.request import urlopen

import numpy as np
import pandas as pd

ROOT: Final = Path(__file__).resolve().parent
DATA_DIR: Final = ROOT / "data"
DEFAULT_CACHE_DIR: Final = ROOT / "cache_data" / "atb"
COST_OUTPUT_NAME: Final = "technology_costs_atb.parquet"
HEAT_RATE_OUTPUT_NAME: Final = "technology_heat_rates_nrelatb.csv"

COST_COLUMNS: Final = [
    "data_year",
    "basis_year",
    "cap_recovery_years",
    "cost_case",
    "financial_case",
    "technology",
    "tech_detail",
    "parameter",
    "parameter_value",
    "units",
    "dollar_year",
]
HEAT_RATE_COLUMNS: Final = [
    "technology",
    "tech_detail",
    "cost_case",
    "basis_year",
    "heat_rate",
    "data_year",
]
COST_KEY_COLUMNS: Final = [
    column for column in COST_COLUMNS if column not in {"parameter_value", "units"}
]
HEAT_RATE_KEY_COLUMNS: Final = [
    column for column in HEAT_RATE_COLUMNS if column != "heat_rate"
]

CORE_METRICS_2023: Final = ["WACC Real", "CAPEX", "Fixed O&M", "Variable O&M", "Heat Rate"]
CORE_METRICS_2024_PLUS: Final = CORE_METRICS_2023 + ["OCC", "GCC", "CFC"]
CORE_METRIC_MAP: Final = {
    "WACC Real": "wacc_real",
    "CAPEX": "capex_mw",
    "Fixed O&M": "fixed_o_m_mw",
    "Variable O&M": "variable_o_m_mwh",
    "Heat Rate": "heat_rate",
}

BATTERY_OM_RATE: Final = 0.025
BATTERY_CAP_RECOVERY_YEARS: Final = 15
BATTERY_SCENARIOS: Final = ("Advanced", "Moderate", "Conservative")
BATTERY_ENERGY_MARKER: Final = "Battery Energy Capital Cost ($/kWh)"
BATTERY_POWER_MARKER: Final = "Battery Power Capital Cost ($/kW)"
REQUIRED_BATTERY_PARAMETERS: Final = {
    "capex_mw",
    "capex_mwh",
    "fixed_o_m_mw",
    "fixed_o_m_mwh",
    "variable_o_m_mwh",
}

# Years whose committed rows may be reused when the official source can no
# longer reproduce them. ATB 2025 is the new release and is always required.
FALLBACK_YEARS: Final = {2023, 2024}

YEAR_SETTINGS: Final = {
    2023: {
        "parquet_url": (
            "https://oedi-data-lake.s3.amazonaws.com/ATB/electricity/parquet/2023/ATBe.parquet"
        ),
        "workbook_url": (
            "https://data.openei.org/files/5865/2023-ATB-Data_Master_v9.0.xlsx"
        ),
        "core_metrics": CORE_METRICS_2023,
        "construction_finance": False,
        # The 2023 build kept heat-rate rows inside the cost table.
        "keep_heat_rate_in_costs": True,
        "financial_cases": None,
        "battery_technology": "Battery",
        "battery_tech_detail": "*",
        "battery_financial_case": "Market",
    },
    2024: {
        "parquet_url": (
            "https://oedi-data-lake.s3.amazonaws.com/ATB/electricity/parquet/2024/v4.0.0/ATBe.parquet"
        ),
        "workbook_url": "https://data.openei.org/files/6006/2024_v3_Workbook.xlsx",
        "core_metrics": CORE_METRICS_2024_PLUS,
        "construction_finance": True,
        "keep_heat_rate_in_costs": False,
        "financial_cases": None,
        "battery_technology": "Utility-Scale Battery Storage",
        "battery_tech_detail": "Lithium Ion",
        "battery_financial_case": "Market",
    },
    2025: {
        "parquet_url": (
            "https://oedi-data-lake.s3.amazonaws.com/ATB/electricity/parquet/2025/v1.0.0/ATBe.parquet"
        ),
        "workbook_url": (
            "https://data.openei.org/files/8759/2025_v1_Workbook_08-27-2026.xlsx"
        ),
        "core_metrics": CORE_METRICS_2024_PLUS,
        "construction_finance": True,
        "keep_heat_rate_in_costs": False,
        # ATB 2025 renamed the financial cases to R&D / R&D + TC / Exp /
        # Exp + TC. Tax-credit and experience-curve cases are excluded.
        "financial_cases": ("R&D",),
        "battery_technology": "Utility-Scale Battery Storage",
        "battery_tech_detail": "Lithium Ion",
        "battery_financial_case": "R&D",
    },
}


class HistoricalSourceUnavailable(RuntimeError):
    """No committed rows exist to fall back on for a historical ATB year."""


def download(url: str, destination: Path) -> Path:
    """Download *url* once, keeping a reusable local cache."""
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(url) as response, temporary.open("wb") as file:
            shutil.copyfileobj(response, file)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def source_paths(year: int, cache_dir: Path) -> tuple[Path, Path]:
    """Return cached (tidy parquet, workbook) paths for *year*, downloading if needed."""
    settings = YEAR_SETTINGS[year]
    return (
        download(settings["parquet_url"], cache_dir / str(year) / "ATBe.parquet"),
        download(settings["workbook_url"], cache_dir / str(year) / "workbook.xlsx"),
    )


def load_tidy(parquet: Path, year: int) -> pd.DataFrame:
    """Load the tidy ATB summary data with the historical column handling."""
    required = [
        "atb_year",
        "core_metric_parameter",
        "core_metric_case",
        "crpyears",
        "technology",
        "techdetail",
        "scenario",
        "core_metric_variable",
        "value",
    ]
    raw = pd.read_parquet(parquet)
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"ATB {year} tidy data is missing columns: {missing}")
    data = raw.loc[:, required].copy()
    if "techdetail2" in raw.columns:
        empty = data["techdetail"].eq("")
        data.loc[empty, "techdetail"] = raw.loc[empty, "techdetail2"]
    return data


def _construction_finance_factor(tidy: pd.DataFrame, year: int) -> pd.DataFrame:
    """Infer CFF and strip GCC from CAPEX for ATB 2024+ tidy records.

    ``CFF = CAPEX / (OCC + GCC)``; the source CAPEX is preserved as
    ``ATB_CAPEX`` and ``CAPEX`` is recalculated without grid connection
    costs (``ATB_CAPEX - GCC * CFF``).
    """
    index = [
        "atb_year",
        "core_metric_case",
        "crpyears",
        "technology",
        "techdetail",
        "scenario",
        "core_metric_variable",
    ]
    wide = (
        tidy.drop_duplicates(index + ["core_metric_parameter"])
        .pivot(index=index, columns="core_metric_parameter", values="value")
        .reset_index()
        .fillna({column: 0 for column in ("OCC", "GCC", "CAPEX")})
    )
    for column in ("OCC", "GCC", "CAPEX"):
        if column not in wide.columns:
            raise ValueError(
                f"ATB {year} tidy data lacks {column}, which is required to infer "
                "the construction finance factor"
            )
    wide["CFF"] = wide["CAPEX"] / (wide["OCC"] + wide["GCC"])
    wide["ATB_CAPEX"] = wide["CAPEX"]
    wide.loc[wide["CAPEX"] > 0, "CAPEX"] = wide["ATB_CAPEX"] - wide["GCC"] * wide["CFF"]
    melted = wide.melt(id_vars=index, var_name="core_metric_parameter", value_name="value")
    return melted.dropna(subset=["value"])


def normalize_costs(tidy: pd.DataFrame, year: int, settings: dict) -> pd.DataFrame:
    """Normalize filtered tidy ATB records to the committed cost schema.

    *tidy* must already be filtered to the year's core metric parameters
    (and, from 2024 on, have had the construction finance factor applied).
    """
    data = tidy.loc[~tidy["technology"].str.contains("Battery", case=False, na=False)]
    data = data.loc[data["technology"].ne("AEO")]
    if data.empty:
        raise ValueError(f"ATB {year} has no usable non-battery records")
    data = data.rename(
        columns={
            "crpyears": "cap_recovery_years",
            "scenario": "cost_case",
            "core_metric_case": "financial_case",
            "core_metric_variable": "basis_year",
            "value": "parameter_value",
            "core_metric_parameter": "parameter",
            "techdetail": "tech_detail",
            "atb_year": "data_year",
        }
    )
    data["technology"] = data["technology"].str.replace("_FE", "", regex=False)
    if year >= 2024:
        data["tech_detail"] = (
            data["tech_detail"]
            .str.replace("NG ", "", regex=False)
            .str.replace("Coal-", "", regex=False)
        )
    data["data_year"] = year
    data["basis_year"] = pd.to_numeric(data["basis_year"], errors="raise").astype(int)
    data["cap_recovery_years"] = pd.to_numeric(
        data["cap_recovery_years"], errors="raise"
    ).astype(int)
    data["parameter_value"] = pd.to_numeric(data["parameter_value"], errors="raise")
    data["dollar_year"] = year - 2
    data["units"] = np.nan
    data = data.sort_values("parameter_value", ascending=False)
    data = data.drop_duplicates(subset=COST_KEY_COLUMNS)
    mapped = data["parameter"].map(CORE_METRIC_MAP)
    data["parameter"] = data["parameter"].where(mapped.isna(), mapped)
    data.loc[data["parameter"].eq("capex_mw"), "parameter_value"] *= 1000
    data.loc[data["parameter"].eq("fixed_o_m_mw"), "parameter_value"] *= 1000
    return data[COST_COLUMNS].reset_index(drop=True)


def _battery_table(excel: pd.ExcelFile, year: int) -> pd.DataFrame:
    """Return the utility-scale battery sheet from an ATB workbook."""
    sheets = [
        sheet
        for sheet in excel.sheet_names
        if "Utility-Scale Battery" in sheet and "PV-Plus" not in sheet
    ]
    if not sheets:
        raise ValueError(f"ATB {year} workbook has no utility-scale battery sheet")
    sheet = next((name for name in sheets if "R&D" in name), sheets[0])
    return excel.parse(sheet_name=sheet, header=None)


def _marker_position(table: pd.DataFrame, marker: str, year: int) -> tuple[int, int]:
    """Locate the (row, column) of *marker* in a raw battery table."""
    mask = table.astype(str).apply(
        lambda column: column.str.contains(marker, regex=False, na=False)
    )
    rows, columns = mask.values.nonzero()
    if len(rows) == 0:
        raise ValueError(f"ATB {year} workbook battery table does not contain {marker!r}")
    return int(rows[0]), int(columns[0])


def _battery_capex_records(
    table: pd.DataFrame, marker: str, parameter: str, year: int
) -> list[dict[str, object]]:
    """Read one workbook cost table (energy or power) into tidy records."""
    marker_row, marker_column = _marker_position(table, marker, year)
    years = pd.to_numeric(table.iloc[marker_row + 1], errors="coerce")
    year_columns = [
        column
        for column, value in years.items()
        if pd.notna(value) and column > marker_column
    ]
    if not year_columns:
        raise ValueError(f"ATB {year} workbook battery table has no projection years")
    records: list[dict[str, object]] = []
    for offset, scenario in enumerate(BATTERY_SCENARIOS):
        scenario_row = marker_row + 2 + offset
        label = table.iat[scenario_row, marker_column]
        if str(label) != scenario:
            raise ValueError(
                f"ATB {year} workbook battery scenarios changed: expected {scenario!r} "
                f"at row {scenario_row}, found {label!r}"
            )
        for column in year_columns:
            value = pd.to_numeric(table.iat[scenario_row, column], errors="coerce")
            if pd.isna(value):
                continue
            records.append(
                {
                    "basis_year": int(years[column]),
                    "cost_case": scenario,
                    "parameter": parameter,
                    "parameter_value": float(value) * 1000,
                }
            )
    return records


def extract_battery_costs(workbook: Path, year: int, settings: dict) -> pd.DataFrame:
    """Extract utility-scale battery costs from an ATB workbook.

    Power ($/kW) and energy ($/kWh) capital costs are converted to $/MW and
    $/MWh; fixed O&M is 2.5% of the matching capex and variable O&M is zero,
    reproducing the committed battery rows.
    """
    excel = pd.ExcelFile(workbook)
    table = _battery_table(excel, year)
    energy = _battery_capex_records(table, BATTERY_ENERGY_MARKER, "capex_mwh", year)
    power = _battery_capex_records(table, BATTERY_POWER_MARKER, "capex_mw", year)
    if not energy or not power:
        raise ValueError(f"ATB {year} workbook did not yield battery power and energy costs")

    records: list[dict[str, object]] = []
    for record in energy + power:
        om_parameter = (
            "fixed_o_m_mwh" if record["parameter"] == "capex_mwh" else "fixed_o_m_mw"
        )
        records.append(record)
        records.append(
            {
                **record,
                "parameter": om_parameter,
                "parameter_value": record["parameter_value"] * BATTERY_OM_RATE,
            }
        )
    for scenario, basis_year in sorted(
        {(record["cost_case"], record["basis_year"]) for record in records}
    ):
        records.append(
            {
                "basis_year": basis_year,
                "cost_case": scenario,
                "parameter": "variable_o_m_mwh",
                "parameter_value": 0.0,
            }
        )

    battery = pd.DataFrame.from_records(records)
    battery["data_year"] = year
    battery["cap_recovery_years"] = BATTERY_CAP_RECOVERY_YEARS
    battery["financial_case"] = settings["battery_financial_case"]
    battery["technology"] = settings["battery_technology"]
    battery["tech_detail"] = settings["battery_tech_detail"]
    battery["units"] = np.nan
    battery["dollar_year"] = year - 2
    return battery[COST_COLUMNS]


def build_year(parquet: Path, workbook: Path, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one ATB year's cost and heat-rate tables from cached sources."""
    settings = YEAR_SETTINGS[year]
    tidy = load_tidy(parquet, year)
    tidy = tidy.loc[tidy["core_metric_parameter"].isin(settings["core_metrics"])]
    cases = settings["financial_cases"]
    if cases is not None:
        tidy = tidy.loc[tidy["core_metric_case"].isin(cases)]
    if tidy.empty:
        raise ValueError(f"ATB {year} tidy data has no records for the requested metrics")
    if settings["construction_finance"]:
        tidy = _construction_finance_factor(tidy, year)

    costs = pd.concat(
        [normalize_costs(tidy, year, settings), extract_battery_costs(workbook, year, settings)],
        ignore_index=True,
    )
    heat_rates = (
        costs.loc[costs["parameter"].eq("heat_rate")]
        .rename(columns={"parameter_value": "heat_rate"})
        .loc[:, HEAT_RATE_COLUMNS]
        .drop_duplicates()
    )
    if not settings["keep_heat_rate_in_costs"]:
        costs = costs.loc[costs["parameter"].ne("heat_rate")]
    return costs[COST_COLUMNS].reset_index(drop=True), heat_rates.reset_index(drop=True)


def fallback_rows(existing: pd.DataFrame, year: int, artifact: str) -> pd.DataFrame:
    """Return the committed rows for *year* so they can be reused verbatim."""
    rows = existing.loc[existing["data_year"].eq(year)]
    if rows.empty:
        raise HistoricalSourceUnavailable(
            f"No committed {artifact} rows for ATB {year} are available for fallback"
        )
    return rows.copy()


def _key_set(frame: pd.DataFrame, columns: list[str]) -> set[tuple]:
    if frame.empty:
        return set()
    return set(map(tuple, frame[columns].itertuples(index=False, name=None)))


def missing_key_count(rebuilt: pd.DataFrame, existing: pd.DataFrame, columns: list[str]) -> int:
    """Count committed logical keys that the rebuilt data no longer provides."""
    return len(_key_set(existing, columns) - _key_set(rebuilt, columns))


def validate_outputs(costs: pd.DataFrame, heat_rates: pd.DataFrame, years: set[int]) -> None:
    """Validate schema, coverage, uniqueness, values, and battery parameters."""
    if list(costs.columns) != COST_COLUMNS:
        raise ValueError("Cost output schema differs from the required PowerGenome schema")
    if list(heat_rates.columns) != HEAT_RATE_COLUMNS:
        raise ValueError("Heat-rate output schema differs from the required PowerGenome schema")
    if set(costs["data_year"].unique()) != years:
        raise ValueError("Cost output years do not match the requested ATB years")
    if not heat_rates.empty and set(heat_rates["data_year"].unique()) != years:
        raise ValueError("Heat-rate output years do not match the requested ATB years")
    if costs["parameter_value"].isna().any() or not np.isfinite(costs["parameter_value"]).all():
        raise ValueError("Cost output contains missing or non-finite values")
    if not heat_rates.empty and (
        heat_rates["heat_rate"].isna().any() or not np.isfinite(heat_rates["heat_rate"]).all()
    ):
        raise ValueError("Heat-rate output contains missing or non-finite values")
    if costs["tech_detail"].isna().any() or costs["tech_detail"].eq("").any():
        raise ValueError("Cost output contains missing technology details")
    if costs.duplicated(COST_KEY_COLUMNS).any():
        raise ValueError("Cost output contains duplicate logical records")
    if heat_rates.duplicated(HEAT_RATE_KEY_COLUMNS).any():
        raise ValueError("Heat-rate output contains duplicate logical records")
    for year in sorted(years):
        battery = costs.loc[
            (costs["data_year"] == year) & costs["technology"].str.contains("Battery", case=False)
        ]
        parameters = set(battery["parameter"])
        missing = REQUIRED_BATTERY_PARAMETERS - parameters
        if missing:
            raise ValueError(f"ATB {year} battery output misses required parameters: {missing}")


def _read_existing(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cost_path = output_dir / COST_OUTPUT_NAME
    heat_rate_path = output_dir / HEAT_RATE_OUTPUT_NAME
    existing_costs = (
        pd.read_parquet(cost_path)
        if cost_path.exists()
        else pd.DataFrame(columns=COST_COLUMNS)
    )
    existing_heat_rates = (
        pd.read_csv(heat_rate_path)
        if heat_rate_path.exists()
        else pd.DataFrame(columns=HEAT_RATE_COLUMNS)
    )
    return existing_costs, existing_heat_rates


def _reuse_committed_on_coverage_gap(
    costs: pd.DataFrame,
    heat_rates: pd.DataFrame,
    existing_costs: pd.DataFrame,
    existing_heat_rates: pd.DataFrame,
    year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reuse committed rows when the official source lost committed records.

    NREL revises hosted ATB files in place. If the rebuilt data no longer
    contains logical records present in the committed output, the committed
    rows are reused verbatim for both artifacts, keeping every data_year's
    rows tied to a single ATB vintage.
    """
    committed_costs = (
        existing_costs.loc[existing_costs["data_year"].eq(year)]
        if not existing_costs.empty
        else existing_costs
    )
    committed_heat_rates = (
        existing_heat_rates.loc[existing_heat_rates["data_year"].eq(year)]
        if not existing_heat_rates.empty
        else existing_heat_rates
    )
    cost_missing = missing_key_count(costs, committed_costs, COST_KEY_COLUMNS)
    heat_rate_missing = missing_key_count(
        heat_rates, committed_heat_rates, HEAT_RATE_KEY_COLUMNS
    )
    if not cost_missing and not heat_rate_missing:
        return costs, heat_rates
    logging.warning(
        "ATB %s official sources no longer provide %s cost and %s heat-rate "
        "records present in the committed data; reusing committed %s and %s "
        "rows for %s",
        year,
        cost_missing,
        heat_rate_missing,
        COST_OUTPUT_NAME,
        HEAT_RATE_OUTPUT_NAME,
        year,
    )
    return (
        fallback_rows(existing_costs, year, COST_OUTPUT_NAME),
        fallback_rows(existing_heat_rates, year, HEAT_RATE_OUTPUT_NAME),
    )


def build(
    years: tuple[int, ...] = (2023, 2024, 2025),
    cache_dir: Path = DEFAULT_CACHE_DIR,
    output_dir: Path = DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build *years* and atomically replace the committed ATB outputs."""
    existing_costs, existing_heat_rates = _read_existing(output_dir)
    costs_by_year: list[pd.DataFrame] = []
    heat_rates_by_year: list[pd.DataFrame] = []

    for year in years:
        try:
            parquet, workbook = source_paths(year, cache_dir)
            costs, heat_rates = build_year(parquet, workbook, year)
        except Exception as error:
            if year not in FALLBACK_YEARS:
                raise RuntimeError(f"Unable to build required ATB {year} data") from error
            logging.warning(
                "Unable to rebuild ATB %s from official sources (%s); reusing committed rows",
                year,
                error,
            )
            costs = fallback_rows(existing_costs, year, COST_OUTPUT_NAME)
            heat_rates = fallback_rows(existing_heat_rates, year, HEAT_RATE_OUTPUT_NAME)
        else:
            if year in FALLBACK_YEARS:
                costs, heat_rates = _reuse_committed_on_coverage_gap(
                    costs, heat_rates, existing_costs, existing_heat_rates, year
                )
            logging.info(
                "Built ATB %s: %s cost rows, %s heat-rate rows", year, len(costs), len(heat_rates)
            )
        costs_by_year.append(costs)
        heat_rates_by_year.append(heat_rates)

    costs = pd.concat(costs_by_year, ignore_index=True)
    heat_rates = pd.concat(heat_rates_by_year, ignore_index=True)
    for frame, columns in ((costs, COST_COLUMNS), (heat_rates, HEAT_RATE_COLUMNS)):
        for column in ("basis_year", "data_year"):
            if column in columns:
                frame[column] = frame[column].astype(int)
    if "cap_recovery_years" in costs.columns:
        costs["cap_recovery_years"] = costs["cap_recovery_years"].astype(int)
    if "dollar_year" in costs.columns:
        costs["dollar_year"] = costs["dollar_year"].astype(int)
    costs = costs.sort_values(COST_COLUMNS[:-1], kind="mergesort").reset_index(drop=True)
    heat_rates = heat_rates.sort_values(HEAT_RATE_COLUMNS, kind="mergesort").reset_index(drop=True)
    validate_outputs(costs, heat_rates, set(years))

    output_dir.mkdir(parents=True, exist_ok=True)
    cost_tmp = output_dir / Path(COST_OUTPUT_NAME).with_suffix(".tmp.parquet").name
    heat_rate_tmp = output_dir / Path(HEAT_RATE_OUTPUT_NAME).with_suffix(".tmp.csv").name
    costs.to_parquet(cost_tmp, index=False)
    heat_rates.to_csv(heat_rate_tmp, index=False)
    cost_tmp.replace(output_dir / COST_OUTPUT_NAME)
    heat_rate_tmp.replace(output_dir / HEAT_RATE_OUTPUT_NAME)
    return costs, heat_rates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=(2023, 2024, 2025),
        choices=sorted(YEAR_SETTINGS),
        help="ATB data years to build (default: 2023 2024 2025)",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    costs, heat_rates = build(tuple(args.years), args.cache_dir, args.output_dir)
    logging.info(
        "Wrote %s cost rows and %s heat-rate rows to %s",
        len(costs),
        len(heat_rates),
        args.output_dir,
    )


if __name__ == "__main__":
    main()
