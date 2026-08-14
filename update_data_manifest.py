#!/usr/bin/env python3
"""Build or update the versioned provenance manifest for the ``data/`` directory.

The manifest (``data/manifest.json``) records, for every file in ``data/``:

* ``sources`` -- a list of ``{source, source_url}`` objects describing where the data
  came from. A file may rely on one or many upstream sources (e.g. the ReEDS generator
  database *and* ``county2zone.csv``). Human-maintained; the tool preserves edits made
  here across runs.
* ``md5`` -- content hash of the file.
* ``version`` and ``last_updated`` -- ISO date (``YYYY-MM-DD``) of the last change.
* ``history`` -- append-only record of prior ``{version, last_updated, md5}`` entries.

It also tracks two top-level versions:

* ``manifest_version`` -- schema version of the manifest itself; fixed at 1.
* ``data_version`` -- the overall dataset version in calendar-versioning (CalVer)
  format ``YYYY.MM.DD``; advances whenever any file is added or changes.

Run from the repo root:

    uv run python update_data_manifest.py
    uv run python update_data_manifest.py --dry-run
    uv run python update_data_manifest.py --date 2026-08-11
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_VERSION = 1

# Seed provenance for files that have never been seen before. Each file maps to a list of
# {source, source_url} objects (one per upstream). The tool preserves any edits made in
# data/manifest.json on later runs. Source values marked "Unknown - document me" are
# placeholders the maintainer should fill in. A source_url is optional per source.
SOURCES: dict[str, list[dict[str, str]]] = {
    "cpi_data.csv": [
        {
            "source": (
                "U.S. CPI-U (All Urban Consumers, All Items, NSA), annual averages via "
                "FRED with BLS public API fallback (superseded by "
                "dollar_year_adjustment.csv)."
            ),
            "source_url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCNS",
        },
    ],
    "dollar_year_adjustment.csv": [
        {
            "source": (
                "U.S. Implicit Price Deflator for Gross Domestic Product (GDP-IPD, index "
                "2017=100), annual, from FRED with BEA NIPA API fallback."
            ),
            "source_url": (
                "https://fred.stlouisfed.org/graph/fredgraph.csv?id=A191RD3A086NBEA"
            ),
        },
    ],
    "distributed_capacity.parquet": [
        {
            "source": (
                "ReEDS distributed PV capacity by county (stscen2023_mid_case scenario), "
                "aggregated county -> BA and converted DC to AC (ILR 1.1) by "
                "build_new_pg_dg_inputs.py."
            ),
            "source_url": (
                "https://raw.githubusercontent.com/NREL/ReEDS-2.0/main/inputs/"
                "dgen_model_inputs/stscen2023_mid_case/distpvcap_stscen2023_mid_case.csv"
            ),
        },
        {
            "source": (
                "ReEDS county-to-zone mapping used to aggregate capacity to balancing areas."
            ),
            "source_url": (
                "https://raw.githubusercontent.com/NREL/ReEDS-2.0/main/inputs/"
                "county2zone.csv"
            ),
        },
    ],
    "distributed_profiles.parquet": [
        {
            "source": (
                "ReEDS distributed PV generation profiles (multi-year "
                "distpv-reference_ba.h5), processed by build_new_pg_dg_inputs.py."
            ),
            "source_url": (
                "https://github.com/NREL/ReEDS-2.0/raw/main/inputs/variability/multi_year/"
                "distpv-reference_ba.h5"
            ),
        },
    ],
    "fuel_prices.parquet": [
        {
            "source": (
                "EIA Annual Energy Outlook (AEO) bulk data: 'Energy Prices : Electric "
                "Power' series by census division / scenario / fuel, mapped to ReEDS "
                "Balancing Areas by .github/skills/eia-fuel-prices."
            ),
            "source_url": "https://www.eia.gov/opendata/bulk/AEO2026.zip",
        },
    ],
    "nerc_reserve_margins.csv": [
        {
            "source": (
                "NERC Long-Term Reliability Assessment (LTRA) 'Reference Margin Level "
                "(%)' by assessment area, extracted by .github/skills/nerc-reserve-margins "
                "and carried forward through 2050."
            ),
            "source_url": (
                "https://www.nerc.com/globalassets/our-work/assessments/nerc_ltra_2025.pdf"
            ),
        },
    ],
    "nerc_reserve_margins_vintage.json": [
        {
            "source": (
                "Sidecar vintage metadata (report year, source URL, retrieval timestamp) "
                "written alongside nerc_reserve_margins.csv by the NERC reserve margins "
                "extractor."
            ),
        },
    ],
    "operational_constraints_reeds.csv": [
        {
            "source": (
                "ReEDS plant characterization module (PCM) defaults (pcm_defaults.json), "
                "transformed by build_operational_constraints_reeds.py into PowerGenome"
                " operational constraints."
            ),
            "source_url": (
                "https://github.com/ReEDS-Model/ReEDS/blob/main/inputs/"
                "plant_characteristics/pcm_defaults.json"
            ),
        },
    ],
    "plant_region_map.csv": [
        {
            "source": (
                "ReEDS generator database (ReEDS_generator_database_final_EIA-NEMS.csv) "
                "as the plant list, joined to ReEDS zones by transform_reeds_generators.py."
            ),
            "source_url": (
                "https://raw.githubusercontent.com/NREL/ReEDS-2.0/refs/heads/main/inputs/"
                "capacity_exogenous/ReEDS_generator_database_final_EIA-NEMS.csv"
            ),
        },
        {
            "source": (
                "ReEDS county-to-zone mapping (county2zone.csv) used to assign each plant "
                "to a model region."
            ),
            "source_url": (
                "https://raw.githubusercontent.com/NREL/ReEDS-2.0/refs/heads/main/inputs/"
                "county2zone.csv"
            ),
        },
    ],
    "reeds_generators_transformed.csv": [
        {
            "source": (
                "ReEDS generator database (ReEDS_generator_database_final_EIA-NEMS.csv) "
                "cleaned and transformed by transform_reeds_generators.py."
            ),
            "source_url": (
                "https://raw.githubusercontent.com/NREL/ReEDS-2.0/refs/heads/main/inputs/"
                "capacity_exogenous/ReEDS_generator_database_final_EIA-NEMS.csv"
            ),
        },
        {
            "source": (
                "PUDL/EIA yearly generators parquet, used for retirement info on existing "
                "units."
            ),
            "source_url": (
                "https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop/nightly/"
                "out_eia__yearly_generators.parquet"
            ),
        },
    ],
    "reeds_load_transformed.parquet": [
        {
            "source": (
                "ReEDS hourly load HDF5 (EER_IRAlow_load_hourly.h5, LFS-backed), filtered "
                "to weather years 2007-2013 and model years 2020-2050 by "
                "transform_reeds_load.py."
            ),
            "source_url": (
                "https://github.com/NREL/ReEDS-2.0/raw/main/inputs/load/"
                "EER_IRAlow_load_hourly.h5"
            ),
        },
    ],
    "regional_cost_multipliers.csv": [
        {
            "source": (
                "EIA capital cost AEO2025 PDF tables mapped to PowerGenome "
                "technologies/regions by extract_location_variation.py."
            ),
            "source_url": (
                "https://www.eia.gov/analysis/studies/powerplants/capitalcost/pdf/"
                "capital_cost_AEO2025.pdf"
            ),
        },
    ],
    "reserve_margins.csv": [
        {
            "source": (
                "ReEDS planning reserve margins (prm_annual.csv) joined to NERC regions "
                "via the ReEDS hierarchy by build_reserve_margins.py."
            ),
            "source_url": (
                "https://raw.githubusercontent.com/ReEDS-Model/ReEDS/main/inputs/reserves/"
                "prm_annual.csv"
            ),
        },
    ],
    "technology_costs_atb.parquet": [
        {
            "source": ("NREL ATB 2024 v4"),
            "source_url": "https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=ATB%2Felectricity%2Fcsv%2F2024%2Fv4.0.0%2F&limit=50",
        },
    ],
    "technology_heat_rates_nrelatb.csv": [
        {
            "source": ("NREL ATB 2024 v4"),
            "source_url": "https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=ATB%2Felectricity%2Fcsv%2F2024%2Fv4.0.0%2F&limit=50",
        },
    ],
    "transmission_capacity_reeds.csv": [
        {
            "source": (
                "ReEDS AC transmission capacity files merged by "
                "merge_transmission_capacity.py."
            ),
            "source_url": (
                "https://raw.githubusercontent.com/NREL/ReEDS-2.0/main/inputs/"
                "transmission/transmission_capacity_init_AC_ba_NARIS2024.csv"
            ),
        },
        {
            "source": (
                "ReEDS non-AC transmission capacity files merged by "
                "merge_transmission_capacity.py."
            ),
            "source_url": (
                "https://raw.githubusercontent.com/NREL/ReEDS-2.0/main/inputs/"
                "transmission/transmission_capacity_init_nonAC_ba.csv"
            ),
        },
    ],
}


def md5_of(path: Path) -> str:
    """Return the lowercase hex MD5 of a file's contents."""
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def calver(date_str: str) -> str:
    """Convert an ISO date (YYYY-MM-DD) to CalVer (YYYY.MM.DD)."""
    return date_str.replace("-", ".")


def _normalize_sources(raw_sources: list[dict]) -> list[dict]:
    """Return a clean list of {source[, source_url]} objects."""
    out = []
    for s in raw_sources:
        item: dict = {"source": s.get("source", "Unknown - document me")}
        if s.get("source_url"):
            item["source_url"] = s["source_url"]
        out.append(item)
    return out


def _prior_sources(prior: dict) -> list[dict]:
    """Extract sources from a prior entry, migrating the old single-field format."""
    if "sources" in prior:
        return _normalize_sources(prior["sources"])
    # Old (pre-2) format: a single ``source`` str plus an optional ``source_url`` str.
    item: dict = {"source": prior.get("source", "Unknown - document me")}
    if prior.get("source_url"):
        item["source_url"] = prior["source_url"]
    return [item]


def build_new_entry(filename: str, md5: str, date_str: str) -> dict:
    """Create a fresh manifest entry for a file never seen before."""
    seed = SOURCES.get(filename)
    if seed is None:
        seed = [{"source": "Unknown - document me"}]
    return {
        "sources": _normalize_sources(seed),
        "last_updated": date_str,
        "version": date_str,
        "md5": md5,
        "history": [],
    }


def advance_entry(prior: dict, md5: str, date_str: str) -> dict:
    """Push the prior entry onto history and create a new current entry for a changed file."""
    history = list(prior.get("history", []))
    history.append(
        {
            "version": prior["version"],
            "last_updated": prior["last_updated"],
            "md5": prior["md5"],
        }
    )
    return {
        "sources": _prior_sources(prior),
        "last_updated": date_str,
        "version": date_str,
        "md5": md5,
        "history": history,
    }


def _migrate_entry(prior: dict) -> dict:
    """Migrate an entry from the old ``source``/``source_url`` format to ``sources``."""
    if "sources" in prior:
        return prior
    entry = dict(prior)
    entry["sources"] = _prior_sources(prior)
    entry.pop("source", None)
    entry.pop("source_url", None)
    return entry


def load_existing(manifest_path: Path) -> dict:
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return {}


def write_manifest(manifest_path: Path, manifest: dict) -> None:
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def update_manifest(
    data_dir: Path, manifest_path: Path, date_str: str, dry_run: bool = False
) -> None:
    existing = load_existing(manifest_path)
    old_files = existing.get("files", {})

    files: dict[str, dict] = {}
    changed_any = False
    for path in sorted(data_dir.iterdir()):
        if not path.is_file() or path.name == manifest_path.name:
            continue
        filename = path.name
        md5 = md5_of(path)
        prior = old_files.get(filename)

        if prior is None:
            files[filename] = build_new_entry(filename, md5, date_str)
            print(f"[added]    {filename}")
            changed_any = True
        elif prior.get("md5") == md5:
            files[filename] = _migrate_entry(prior)
            print(f"[unchanged] {filename}")
        else:
            files[filename] = advance_entry(prior, md5, date_str)
            print(f"[changed]  {filename}")
            changed_any = True

    # Drop files that no longer exist in data/.
    for filename in sorted(set(old_files) - set(files)):
        print(f"[removed]  {filename} (no longer in {data_dir.name}/)")
        changed_any = True

    if changed_any:
        data_version = calver(date_str)
    else:
        data_version = existing.get("data_version", calver(date_str))

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "data_version": data_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }

    if dry_run:
        print(f"\n[dry-run]  would write {manifest_path} (data_version={data_version})")
        return
    write_manifest(manifest_path, manifest)
    print(f"\nWrote {manifest_path} (data_version={data_version})")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build/update data/manifest.json provenance + version history."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory to scan (default: data/)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest.json"),
        help="Path to the manifest file (default: data/manifest.json)",
    )
    parser.add_argument(
        "--date",
        default=datetime.now(timezone.utc).date().isoformat(),
        help="Change date as YYYY-MM-DD (default: UTC today)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing the manifest",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    update_manifest(args.data_dir, args.manifest, args.date, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
