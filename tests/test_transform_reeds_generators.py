import pandas as pd
import pytest

from transform_reeds_generators import create_plant_region_map


def test_plant_map_uses_county_in_each_zone_set():
    generators = pd.DataFrame({"T_PID": [4, 4, 7, 9], "FIPS": ["p01001", "p01001", "01003", "06001"]})
    zone_maps = {
        "z134": pd.DataFrame({"FIPS": [1001, 1003, 6001], "r": ["p1", "p1", "p2"]}),
        "z90": pd.DataFrame({"FIPS": ["01001", "p01003"], "r": ["AL_N", "AL_S"]}),
    }

    result = create_plant_region_map(generators, zone_maps)

    assert result.columns.tolist() == ["plant_id", "region", "region_z90"]
    assert result["plant_id"].tolist() == [4, 7, 9]
    assert result["region"].tolist() == ["p1", "p1", "p2"]
    assert result["region_z90"].iloc[:2].tolist() == ["AL_N", "AL_S"]
    assert pd.isna(result["region_z90"].iloc[2])


def test_plant_map_rejects_duplicate_counties_after_normalization():
    generators = pd.DataFrame({"T_PID": [4], "FIPS": ["p01001"]})
    zone_maps = {
        "z134": pd.DataFrame({"FIPS": [1001], "r": ["p1"]}),
        "z90": pd.DataFrame({"FIPS": ["1001", "01001"], "r": ["AL_N", "AL_S"]}),
    }

    with pytest.raises(pd.errors.MergeError):
        create_plant_region_map(generators, zone_maps)
