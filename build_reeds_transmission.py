#!/usr/bin/env python3
"""Build z134 and z90 ReEDS transmission capacity and route costs."""

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile

import geopandas as gpd
import pandas as pd

from build_reeds_load import REEDS_INPUTS, ZONE_SETS, fetch

TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER2022/COUNTY/tl_2022_us_county.zip"
COLUMNS = ["region_from", "region_to", "firm_ttc_mw", "notes", "zone_set"]


def ac_rows(interfaces, hashes, itls):
    pairs = interfaces.interface.str.split("~~", expand=True)
    pairs.columns = ["r", "rr"]
    pairs["md5_from"] = pairs.r.map(hashes)
    pairs["md5_to"] = pairs.rr.map(hashes)
    result = pairs.merge(itls, on=["md5_from", "md5_to"], how="left", validate="1:1")
    limits = result[["MW_forward", "MW_reverse"]]
    if limits.isna().any().any() or (limits < 0).any().any():
        raise ValueError("Missing or negative native AC transfer limits")
    return result.assign(MW=limits.mean(axis=1), notes="")[["r", "rr", "MW", "notes"]]


def hvdc_rows(lines, counties, county_zones):
    counties = counties[["GEOID", "geometry"]].copy()
    counties["r"] = counties.GEOID.map(county_zones)
    for side, zone in [("from", "r"), ("to", "rr")]:
        endpoints = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(lines[f"{side}_lon"], lines[f"{side}_lat"]),
            crs="EPSG:4326", index=lines.index,
        ).to_crs(counties.crs)
        matches = gpd.sjoin(endpoints, counties.dropna(subset=["r"]), how="left")
        if matches.index.has_duplicates or matches.r.isna().any():
            raise ValueError(f"Unmapped or ambiguous existing HVDC {side} endpoint")
        lines = lines.assign(**{zone: matches.r})
    return lines.loc[lines.r != lines.rr, ["r", "rr", "MW", "name"]].rename(
        columns={"name": "notes"}
    )


def combine_capacity(ac, hvdc, b2b, zones, zone_set):
    rows = pd.concat([ac, hvdc, b2b.assign(notes="B2B converter")], ignore_index=True)
    if not set(rows.r).union(rows.rr) <= set(zones) or (rows.r == rows.rr).any():
        raise ValueError("Invalid transmission endpoints")
    if rows.MW.isna().any() or (rows.MW < 0).any():
        raise ValueError("Missing or negative initial capacity")
    rows["region_from"] = rows[["r", "rr"]].min(axis=1)
    rows["region_to"] = rows[["r", "rr"]].max(axis=1)
    return rows.groupby(["region_from", "region_to"], as_index=False).agg(
        firm_ttc_mw=("MW", "sum"),
        notes=("notes", lambda values: "; ".join(dict.fromkeys(filter(None, values)))),
    ).assign(zone_set=zone_set)[COLUMNS]


def write_csv(path, frame):
    with NamedTemporaryFile(dir=path.parent, suffix=".csv", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        frame.to_csv(temporary, index=False)
        temporary.chmod(path.stat().st_mode & 0o777 if path.exists() else 0o644)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def expansion_costs(routes, hashes, dollar_year, zone_set):
    zones = pd.Series(hashes.index, index=hashes.array)
    routes = routes.assign(r=routes.start.map(zones), rr=routes.end.map(zones))
    routes = routes.dropna(subset=["r", "rr"]).copy()
    routes = routes.rename(columns={"r": "region_from", "rr": "region_to", "voltage": "voltage_kv"})
    keys = ["region_from", "region_to", "polarity", "voltage_kv"]
    if routes.duplicated(keys).any() or (routes.region_from == routes.region_to).any():
        raise ValueError("Invalid or duplicated route costs")
    measures = routes[["cost_MUSD", "length_miles"]]
    if measures.isna().any().any() or (measures <= 0).any().any():
        raise ValueError("Missing or nonpositive route cost/distance")
    return routes.assign(
        zone_set=zone_set, dollar_year=dollar_year,
        capital_cost_usd=routes.cost_MUSD.mul(1_000_000).round(2),
    ).sort_values(keys)[["zone_set", *keys, "capital_cost_usd", "dollar_year", "length_miles"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache_data/reeds-transmission"))
    args = parser.parse_args()

    def read(path, zone_set=None, **kwargs):
        cache = args.cache_dir / zone_set if zone_set else args.cache_dir
        return pd.read_csv(fetch(f"{REEDS_INPUTS}/{path}", cache), **kwargs)

    counties = gpd.read_file(fetch(TIGER_URL, args.cache_dir), columns=["GEOID"])
    lines = read("transmission/hvdc_existing.csv")
    itls = read("transmission/itl_NARIS.csv")
    routes = read("transmission/transmission_cost_distance.csv")
    dollar_year = read("transmission/dollaryear.csv").set_index("filename").loc[
        "transmission_cost_distance.csv", "US_dollar_year"
    ]
    capacity, costs = [], []
    for zone_set, count in ZONE_SETS.items():
        prefix = f"zones/{zone_set}"
        hashes = read(f"{prefix}/zonehash.csv", zone_set).set_index("r").md5
        if len(hashes) != count or not hashes.index.is_unique or not hashes.is_unique:
            raise ValueError(f"Invalid official {zone_set} zone hashes")
        ac = ac_rows(read(f"{prefix}/interfaces_r.csv", zone_set), hashes, itls)
        county_zones = read(f"{prefix}/county2zone.csv", zone_set, dtype={"FIPS": str}).set_index("FIPS").r
        hvdc = hvdc_rows(lines, counties, county_zones)
        b2b = read(f"{prefix}/b2b.csv", zone_set)
        capacity.append(combine_capacity(ac, hvdc, b2b, hashes.index, zone_set))
        costs.append(expansion_costs(routes, hashes, dollar_year, zone_set))
        print(f"{zone_set}: {len(capacity[-1])} capacity rows and {len(costs[-1])} expansion routes")
    write_csv(args.data_dir / "transmission_capacity_reeds.csv", pd.concat(capacity))
    write_csv(args.data_dir / "transmission_expansion_cost_reeds.csv", pd.concat(costs))


if __name__ == "__main__":
    main()
