import json
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

import match_reeds_regions_to_cities as matcher


def test_geocoding_builds_geometry_from_cached_and_live_coordinates(tmp_path, monkeypatch):
    cache = tmp_path / "cities.json"
    cache.write_text(json.dumps({"Richmond, Virginia": {"latitude": 37.5, "longitude": -77.5}}))
    monkeypatch.setattr(matcher, "GEOCODE_CACHE_FILE", cache)
    geocoder = Mock()
    geocoder.geocode.return_value = SimpleNamespace(latitude=35.8, longitude=-78.6)
    monkeypatch.setattr(matcher, "Nominatim", Mock(return_value=geocoder))
    cities = pd.DataFrame({"city": ["Richmond", "Raleigh"], "state": ["Virginia", "North Carolina"]})

    result = matcher.geocode_cities(cities, delay=0)

    assert list(result.geometry.x) == [-77.5, -78.6]
    assert list(result.geometry.y) == [37.5, 35.8]
    assert result.crs.to_epsg() == 4326
    geocoder.geocode.assert_called_once_with("Raleigh, North Carolina, USA")
