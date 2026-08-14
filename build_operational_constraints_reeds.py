#!/usr/bin/env python3
"""Build data/operational_constraints_reeds.csv from ReEDS pcm_defaults.json.

This script regenerates the operational constraints table used by PowerGenome
settings (``operational_constraints_table: operational_constraints_reeds.csv``)
from the ReEDS plant characterization module (PCM) defaults file on GitHub:
https://github.com/ReEDS-Model/ReEDS/blob/main/inputs/plant_characteristics/pcm_defaults.json

What it does:
1. Fetches ``pcm_defaults.json`` from the ReEDS repo (or reads a local copy via
   ``--pcm-path``) and turns one row per technology into PowerGenome/GenX-style
   operational constraints columns (Min_Power, Eff_Up, Ramp_Up_Percentage, etc.).
2. Drops all concentrating solar power (CSP) technologies (``csp*``) because
   PowerGenome does not use CSP-specific operational constraints.
3. Applies the ``tech_map``: for each new-build ATB/EIA technology name (key),
   copies the operational constraints of the corresponding existing ReEDS
   technology (value) so new-build resources named after ATB technologies have
   constraint values. The entries flagged ``# <-- USER-PROVIDED`` are the
   mappings supplied by the user; the remaining entries are added by this script
   so that every technology in the ReEDS generators file is covered, and each is
   commented with its reasoning. Mappings that are assumptions are marked
   ``# ASSUMED - verify``.
4. Appends curated rows for existing EIA-860 technology names used in
   ``data/reeds_generators_transformed.csv`` that have no PCM counterpart. These
   were originally hand-curated in the repository file and are preserved here
   verbatim so existing generators keep their historical constraint values.

Output columns (19), matching the original file:
Resource, region, Min_Power, Self_Disch, Eff_Up, Eff_Down, Ramp_Up_Percentage,
Ramp_Dn_Percentage, Up_Time, Down_Time, Existing_Charge_Cap_MW, Max_Cap_MWh,
Min_Cap_MWh, Max_Charge_Cap_MW, Reg_Cost, Rsv_Cost, Start_Cost_per_MW,
Max_Duration, Min_Duration

Re-run from the repo root with:  python build_operational_constraints_reeds.py
"""

import argparse
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PCM_DEFAULTS_URL = (
    "https://raw.githubusercontent.com/ReEDS-Model/ReEDS/main/"
    "inputs/plant_characteristics/pcm_defaults.json"
)
OUTPUT_PATH = Path("data/operational_constraints_reeds.csv")
REQUEST_TIMEOUT_SECONDS = 30

# Map new-build ATB technology names (keys, in the ATB naming convention used by
# PowerGenome, e.g. "naturalgas_2on1_combined_cycle_fframe") to the existing
# ReEDS technology names (values) whose PCM operational constraints they should
# inherit.
TECH_MAP = {
    # --- USER-PROVIDED MAPPINGS (verbatim from the original script) ---
    "naturalgas_2on1_combined_cycle_fframe": "gas-cc",  # <-- USER-PROVIDED
    "naturalgas_1on1_combined_cycle_fframe": "gas-cc",  # <-- USER-PROVIDED
    "naturalgas_2on1_combined_cycle_fframe_95_ccs": "gas-cc-ccs",  # <-- USER-PROVIDED
    "naturalgas_combustion_turbine_fframe": "gas-ct",  # <-- USER-PROVIDED
    "pumped storage hydropower": "pumped-hydro",  # <-- USER-PROVIDED
    "utility-scale_battery_storage": "battery_li",  # <-- USER-PROVIDED
    # --- ADDITIONAL MAPPINGS so all ReEDS generator technologies are covered ---
    # F-Frame CC 97% CCS maps to the existing ReEDS "max" CCS variant.
    "naturalgas_2on1_combined_cycle_fframe_97_ccs": "gas-cc-ccs_max",
    "naturalgas_1on1_combined_cycle_fframe_97_ccs": "gas-cc-ccs_max",
    # H-Frame CC variants map to the matching ReEDS H-frame technologies.
    "naturalgas_2on1_combined_cycle_hframe": "gas-cc_h_2x1",
    "naturalgas_1on1_combined_cycle_hframe": "gas-cc_h_1x1",
    "naturalgas_2on1_combined_cycle_hframe_95_ccs": "gas-cc_h_2x1-ccs_mod",
    "naturalgas_2on1_combined_cycle_hframe_97_ccs": "gas-cc_h_2x1-ccs_max",
    "naturalgas_1on1_combined_cycle_hframe_95_ccs": "gas-cc_h_1x1-ccs_mod",
    "naturalgas_1on1_combined_cycle_hframe_97_ccs": "gas-cc_h_1x1-ccs_max",
    # Combustion turbine aeroderivative and fuel cell new-build ATB techs.
    "naturalgas_combustion_turbine_aero": "gas-ct_aero",
    "naturalgas_fuel_cell": "ng-fuel-cell",
    # Coal and landfill-gas new-build techs. Nuclear, biopower and geothermal
    # are NOT listed here: their ATB names already match PCM row names, so the
    # base PCM rows cover them (adding them here would create duplicates).
    "coal_new": "coal-new",
    "coal_igcc": "coal-igcc",
    "hydropower": "hydd",  # ASSUMED - verify: new-build hydro inherits hydd PCM
    "landfill_gas": "lfill-gas",  # ASSUMED - verify: EIA name for landfill gas
}

# Column order of the output file (matches the historical operational constraints file).
OUTPUT_COLUMNS = [
    "Resource",
    "region",
    "Min_Power",
    "Self_Disch",
    "Eff_Up",
    "Eff_Down",
    "Ramp_Up_Percentage",
    "Ramp_Dn_Percentage",
    "Up_Time",
    "Down_Time",
    "Existing_Charge_Cap_MW",
    "Max_Cap_MWh",
    "Min_Cap_MWh",
    "Max_Charge_Cap_MW",
    "Reg_Cost",
    "Rsv_Cost",
    "Start_Cost_per_MW",
    "Max_Duration",
    "Min_Duration",
]

# Hand-curated rows for existing EIA-860 technology names from
# data/reeds_generators_transformed.csv that do not exist in pcm_defaults.json.
# Preserved verbatim from the repository's historical operational constraints
# file so existing generators keep their established constraint values.
CURATED_ROWS = [
    {
        "Resource": "batteries",
        "Min_Power": np.nan, "Eff_Up": 0.921954446, "Eff_Down": 0.921954446,
        "Up_Time": np.nan, "Down_Time": np.nan, "Start_Cost_per_MW": np.nan,
        "Max_Duration": 8, "Min_Duration": 0,
    },
    {
        "Resource": "coal",
        "Min_Power": 0.4, "Eff_Up": 1, "Eff_Down": 1,
        "Ramp_Up_Percentage": 1.2, "Ramp_Dn_Percentage": 1.2,
        "Up_Time": 24, "Down_Time": 12, "Start_Cost_per_MW": 129,
        "Max_Duration": np.nan, "Min_Duration": np.nan,
    },
    {
        "Resource": "natural gas fired combined cycle",
        "Min_Power": 0.5, "Eff_Up": 1, "Eff_Down": 1,
        "Ramp_Up_Percentage": 3, "Ramp_Dn_Percentage": 3,
        "Up_Time": 6, "Down_Time": 8, "Start_Cost_per_MW": 79,
        "Max_Duration": np.nan, "Min_Duration": np.nan,
    },
    {
        "Resource": "natural gas fired combustion turbine",
        "Min_Power": 0.6, "Eff_Up": 1, "Eff_Down": 1,
        "Ramp_Up_Percentage": 4.8, "Ramp_Dn_Percentage": 4.8,
        "Up_Time": 0, "Down_Time": 0, "Start_Cost_per_MW": 69,
        "Max_Duration": np.nan, "Min_Duration": np.nan,
    },
    {
        "Resource": "conventional hydroelectric",
        "Min_Power": np.nan, "Eff_Up": 1, "Eff_Down": 1,
        "Ramp_Up_Percentage": 3, "Ramp_Dn_Percentage": 3,
        "Up_Time": np.nan, "Down_Time": np.nan, "Start_Cost_per_MW": np.nan,
        "Max_Duration": np.nan, "Min_Duration": np.nan,
    },
    {
        "Resource": "Run of River",
        "Min_Power": 0.7, "Eff_Up": 1, "Eff_Down": 1,
        "Ramp_Up_Percentage": 3, "Ramp_Dn_Percentage": 3,
        "Up_Time": np.nan, "Down_Time": np.nan, "Start_Cost_per_MW": np.nan,
        "Max_Duration": np.nan, "Min_Duration": np.nan,
    },
    {
        "Resource": "Other_peaker",
        "Min_Power": 0.6, "Eff_Up": 1, "Eff_Down": 1,
        "Ramp_Up_Percentage": 3.6, "Ramp_Dn_Percentage": 3.6,
        "Up_Time": 0, "Down_Time": 0, "Start_Cost_per_MW": 69,
        "Max_Duration": np.nan, "Min_Duration": np.nan,
    },
    {
        "Resource": "Natural gas steam turbine",
        "Min_Power": 0.6, "Eff_Up": 1, "Eff_Down": 1,
        "Ramp_Up_Percentage": 3.6, "Ramp_Dn_Percentage": 3.6,
        "Up_Time": 0, "Down_Time": 0, "Start_Cost_per_MW": 69,
        "Max_Duration": np.nan, "Min_Duration": np.nan,
    },
    {
        "Resource": "Hydroelectric Pumped Storage",
        "Min_Power": np.nan, "Eff_Up": 0.894427191, "Eff_Down": 0.894427191,
        "Ramp_Up_Percentage": np.nan, "Ramp_Dn_Percentage": np.nan,
        "Up_Time": np.nan, "Down_Time": np.nan, "Start_Cost_per_MW": np.nan,
        "Max_Duration": 20, "Min_Duration": 8,
    },
]


def fetch_pcm_defaults(url: str) -> pd.DataFrame:
    """Fetch the ReEDS PCM defaults file from GitHub as a tidy DataFrame."""
    response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    df = (
        pd.read_json(StringIO(response.text))
        .T.reset_index()
        .rename(columns={"index": "tech"})
    )
    return df


def read_pcm_defaults(path: Path) -> pd.DataFrame:
    """Read a local copy of pcm_defaults.json as a tidy DataFrame."""
    df = (
        pd.read_json(path)
        .T.reset_index()
        .rename(columns={"index": "tech"})
    )
    return df


def build_operational_constraints(oper_r: pd.DataFrame) -> pd.DataFrame:
    """Build the operational constraints DataFrame from PCM rows.

    Follows the user's original transform: converts PCM fields into
    PowerGenome/GenX columns, applies battery/pumped storage efficiency and
    duration overrides, flips 100% minimum stable levels down to 70%, and
    appends new-build ATB technology rows via ``tech_map`` plus the curated
    existing-technology rows.

    Args:
        oper_r: Tidy DataFrame with a ``tech`` column and PCM fields.

    Returns:
        The operational constraints DataFrame with columns in ``OUTPUT_COLUMNS``.
    """
    oper_pg = pd.DataFrame(
        {
            "Resource": oper_r["tech"],
            "region": ["all"] * len(oper_r),
            "Min_Power": oper_r["min_stable_level_percentage"],
            "Self_Disch": [0] * len(oper_r),
            "Eff_Up": np.ones(len(oper_r)),  # only for battery / storage
            "Eff_Down": np.ones(len(oper_r)),
            "Ramp_Up_Percentage": oper_r["max_ramp_up_percentage"] * 60,
            "Ramp_Dn_Percentage": oper_r["max_ramp_up_percentage"] * 60,
            "Up_Time": oper_r["min_up_time"],
            "Down_Time": oper_r["min_down_time"],
            "Existing_Charge_Cap_MW": [0] * len(oper_r),  # check with Greg
            "Max_Cap_MWh": [-1] * len(oper_r),
            "Min_Cap_MWh": [-1] * len(oper_r),
            "Max_Charge_Cap_MW": [-1] * len(oper_r),
            "Reg_Cost": [0] * len(oper_r),
            "Rsv_Cost": [0] * len(oper_r),
            "Start_Cost_per_MW": oper_r["start_cost_per_MW"],
            "Max_Duration": np.nan,
            "Min_Duration": np.nan,
        }
    )

    # Drop all concentrating solar power (CSP) technologies.
    oper_pg = oper_pg.loc[~oper_pg["Resource"].astype(str).str.startswith("csp")].copy()

    # Battery storage: 85% round trip efficiency, 8h duration, no min duration.
    oper_pg.loc[oper_pg.Resource.str.contains("battery"), "Eff_Up"] = np.sqrt(0.85)
    oper_pg.loc[oper_pg.Resource.str.contains("battery"), "Eff_Down"] = np.sqrt(0.85)
    oper_pg.loc[oper_pg.Resource.str.contains("battery"), "Max_Duration"] = 8
    oper_pg.loc[oper_pg.Resource.str.contains("battery"), "Min_Duration"] = 0

    # Pumped storage: 80% round trip efficiency, 13h max / 11h min duration.
    oper_pg.loc[oper_pg.Resource.str.contains("pumped"), "Max_Duration"] = 13
    oper_pg.loc[oper_pg.Resource.str.contains("pumped"), "Min_Duration"] = 11
    oper_pg.loc[oper_pg.Resource.str.contains("pumped"), "Eff_Up"] = np.sqrt(0.8)
    oper_pg.loc[oper_pg.Resource.str.contains("pumped"), "Eff_Down"] = np.sqrt(0.8)

    # Change all Min_Power values of 1.0 to 0.7.
    oper_pg.loc[oper_pg.Min_Power == 1, "Min_Power"] = 0.7

    # Create copies of existing rows to add new-build ATB/EIA technologies.
    df_list = []
    for new_name, reeds_name in TECH_MAP.items():
        df_new = oper_pg[oper_pg["Resource"] == reeds_name].copy()
        if df_new.empty:
            raise ValueError(
                f"tech_map value '{reeds_name}' (for '{new_name}') has no PCM "
                "row. Check the value against pcm_defaults.json."
            )
        df_new["Resource"] = new_name
        df_list.append(df_new)
    oper_pg = pd.concat([oper_pg] + df_list, ignore_index=True)

    # Append the curated existing-technology rows.
    curated = pd.DataFrame(CURATED_ROWS).reindex(columns=OUTPUT_COLUMNS)
    # Curated rows default to the same fixed fields as PCM rows.
    curated["region"] = "all"
    curated["Self_Disch"] = 0
    curated["Existing_Charge_Cap_MW"] = 0
    curated["Max_Cap_MWh"] = -1
    curated["Min_Cap_MWh"] = -1
    curated["Max_Charge_Cap_MW"] = -1
    curated["Reg_Cost"] = 0
    curated["Rsv_Cost"] = 0
    oper_pg = pd.concat([oper_pg, curated[OUTPUT_COLUMNS]], ignore_index=True)

    # Match the historical file's 9-decimal float formatting for efficiency.
    for col in ("Eff_Up", "Eff_Down"):
        oper_pg[col] = oper_pg[col].round(9)

    # Match the historical file's formatting: render every numeric value as a
    # whole number when it is one (1 not 1.0) and clean float artifacts
    # (0.6000000000000001 -> 0.6). Ramp values keep their 7-decimal products
    # and efficiencies their 9-decimal values.
    numeric = ["Min_Power", "Self_Disch", "Eff_Up", "Eff_Down",
               "Ramp_Up_Percentage", "Ramp_Dn_Percentage", "Up_Time", "Down_Time",
               "Existing_Charge_Cap_MW", "Max_Cap_MWh", "Min_Cap_MWh",
               "Max_Charge_Cap_MW", "Reg_Cost", "Rsv_Cost", "Start_Cost_per_MW",
               "Max_Duration", "Min_Duration"]
    for col in numeric:
        cleaned = []
        for v in oper_pg[col]:
            if pd.isna(v):
                cleaned.append(pd.NA)
            else:
                v = round(float(v), 9)
                cleaned.append(int(v) if v == int(v) else v)
        oper_pg[col] = pd.array(cleaned, dtype="object")

    return oper_pg[OUTPUT_COLUMNS]


def validate_operational_constraints(df: pd.DataFrame) -> None:
    """Validate the operational constraints DataFrame.

    Raises:
        ValueError: If Resource names are duplicated, if any CSP technology
            remains, or if required numeric fields are missing for non-storage,
            non-VRE technologies.
    """
    duplicates = df["Resource"].duplicated()
    if duplicates.any():
        dup_rows = df.loc[duplicates, "Resource"].tolist()
        raise ValueError(
            f"{len(dup_rows)} duplicate Resource name(s) in operational "
            f"constraints: {dup_rows}"
        )

    csp_rows = df.loc[df["Resource"].astype(str).str.startswith("csp"), "Resource"]
    if not csp_rows.empty:
        raise ValueError(
            f"{len(csp_rows)} CSP technology row(s) found after dropping CSP: "
            f"{csp_rows.tolist()}"
        )

    # Thermal technologies must have ramp, up-time, down-time and min power.
    # Rows that legitimately have no min-power / ramp constraints: storage,
    # hydro, and plain import resources (matching how the historical file
    # treats them).
    no_constraint = df["Resource"].astype(str).str.contains(
        r"batter|pumped|hyd|imports|Run of River", case=False, regex=True
    )
    required = [
        "Min_Power",
        "Ramp_Up_Percentage",
        "Ramp_Dn_Percentage",
        "Up_Time",
        "Down_Time",
    ]
    missing_mask = df.loc[~no_constraint, required].isna().any(axis=1)
    bad = df.loc[~no_constraint].loc[missing_mask, "Resource"].tolist()
    if bad:
        raise ValueError(
            f"{len(bad)} thermal-row(s) missing Min_Power/ramp/up-downtime "
            f"values: {bad}"
        )


def build_operational_constraints_reeds(pcm: pd.DataFrame) -> pd.DataFrame:
    """Build and validate the full operational constraints DataFrame."""
    df = build_operational_constraints(pcm)
    validate_operational_constraints(df)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build data/operational_constraints_reeds.csv from ReEDS PCM defaults."
    )
    parser.add_argument(
        "--pcm-path",
        type=Path,
        default=None,
        help="Optional local path to pcm_defaults.json (default: fetch from GitHub).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output CSV path (default: {OUTPUT_PATH}).",
    )
    args = parser.parse_args()

    print("Building operational constraints...")
    if args.pcm_path is not None:
        print(f"Reading PCM defaults from {args.pcm_path}")
        pcm = read_pcm_defaults(args.pcm_path)
    else:
        print(f"Fetching PCM defaults from {PCM_DEFAULTS_URL}")
        pcm = fetch_pcm_defaults(PCM_DEFAULTS_URL)

    constraints = build_operational_constraints_reeds(pcm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    constraints.to_csv(args.output, index=False)
    print(f"Saved {len(constraints)} rows to {args.output}")
    print(constraints.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
