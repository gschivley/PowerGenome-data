"""Offline coverage for immutable input selection, verification and provenance."""

import hashlib
import json
from unittest.mock import Mock

import pytest
import requests

from locked_sources import SourceLock, ZENODO_TEMPLATE, receipt_path
from update_data_manifest import update_manifest

COMMIT = "a" * 40
PAYLOAD = b"verified input"


def checksum(data, algorithm):
    return f"{algorithm}:{hashlib.new(algorithm, data).hexdigest()}"


def github(path="inputs/a.csv", data=PAYLOAD, **changes):
    return {
        "kind": "github",
        "repository": "owner/repo",
        "commit": COMMIT,
        "path": path,
        "checksum": checksum(data, "sha256"),
        **changes,
    }


def zenodo(data=PAYLOAD):
    return {
        "kind": "zenodo", "record_id": 123, "filename": "profiles.h5",
        "checksum": checksum(data, "md5"),
    }


def source_lock(tmp_path, pins):
    path = tmp_path / "test.lock.json"
    path.write_text(json.dumps({"lock_version": 1, "sources": pins}))
    return SourceLock(tmp_path / "cache", path)


def seed(lock, key, data):
    path = lock.cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.fixture
def http(monkeypatch):
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.iter_content.return_value = iter([PAYLOAD[:4], PAYLOAD[4:]])
    client = Mock(return_value=response)
    monkeypatch.setattr("locked_sources.requests.get", client)
    return client, response


@pytest.mark.parametrize(("pin", "expected_url"), [
    (github(), f"https://raw.githubusercontent.com/owner/repo/{COMMIT}/inputs/a.csv"),
    (github(lfs=True), f"https://media.githubusercontent.com/media/owner/repo/{COMMIT}/inputs/a.csv"),
    (zenodo(), "https://zenodo.org/records/123/files/profiles.h5"),
])
def test_downloads_only_the_locked_version(tmp_path, http, pin, expected_url):
    client, _ = http
    lock = source_lock(tmp_path, {"source": pin})
    path = lock.fetch("source")
    assert path.read_bytes() == PAYLOAD
    client.assert_called_once_with(expected_url, stream=True, timeout=(15, 90))
    assert lock.fetch("source") == path
    assert client.call_count == 1


@pytest.mark.parametrize("pin", [
    None,
    github(commit="main"),
    github(path="inputs/../a.csv"),
    github(checksum="sha256:not-a-digest"),
    {**zenodo(), "record_id": "latest"},
    {"kind": "https", "url": "https://zenodo.org/records/123/files/profiles.h5",
     "checksum": checksum(PAYLOAD, "sha256")},
])
def test_missing_or_invalid_pins_fail_before_fetch(tmp_path, http, pin):
    client, _ = http
    lock = source_lock(tmp_path, {"source": pin} if pin is not None else {})
    with pytest.raises(ValueError, match="(?:Missing|Invalid) source pin"):
        lock.fetch("source")
    client.assert_not_called()


def test_cache_separates_paths_commits_and_records(tmp_path):
    lock = source_lock(tmp_path, {
        "first": github("inputs/z134/a.csv"),
        "second": github("inputs/z90/a.csv"),
        "new_commit": github("inputs/z134/a.csv", commit="b" * 40),
        "zenodo": zenodo(),
        "second_record": {**zenodo(), "record_id": 321},
    })
    assert len({lock.cache_path(key) for key in lock.sources}) == 5


def test_bad_cache_and_local_override_are_never_used_or_replaced(tmp_path, http):
    client, _ = http
    lock = source_lock(tmp_path, {"source": github()})
    cache = seed(lock, "source", b"corrupt")
    with pytest.raises(ValueError, match="Checksum mismatch for source"):
        lock.fetch("source")
    with pytest.raises(ValueError, match="Checksum mismatch for source"):
        lock.fetch("source", local_path=cache)
    assert cache.read_bytes() == b"corrupt"
    assert lock.used == {}
    client.assert_not_called()


@pytest.mark.parametrize("network_failure", [False, True])
def test_incomplete_or_corrupt_download_is_not_cached(tmp_path, http, network_failure):
    _, response = http
    lock = source_lock(tmp_path, {"source": github(lfs=True)})

    def chunks():
        yield b"version https://git-lfs.github.com/spec/v1\n"
        if network_failure:
            raise requests.ConnectionError("interrupted")

    response.iter_content.return_value = chunks()
    expected = RuntimeError if network_failure else ValueError
    with pytest.raises(expected, match="source"):
        lock.fetch("source")
    assert list(lock.cache_path("source").parent.iterdir()) == []
    assert lock.used == {}


def catalog_bytes(pin, *, rows=1, **changes):
    row = {
        "filename": pin["filename"], "record_id": pin["record_id"],
        "md5": pin["checksum"].split(":")[1], "url_base": ZENODO_TEMPLATE,
        **changes,
    }
    fields = ("filename", "record_id", "md5", "url_base")
    return (",".join(fields) + "\n" + (",".join(str(row[key]) for key in fields) + "\n") * rows).encode()


@pytest.mark.parametrize("changes", [
    {"rows": 0}, {"rows": 2}, {"record_id": 456}, {"md5": "0" * 32},
    {"url_base": "https://zenodo.org/records/latest/files/{filename}"},
    {"filename": "other.h5"},
])
def test_catalog_must_agree_with_the_exact_zenodo_pin(tmp_path, http, changes):
    client, _ = http
    profiles = zenodo()
    catalog = catalog_bytes(profiles, **changes)
    lock = source_lock(tmp_path, {"catalog": github(data=catalog), "profiles": profiles})
    seed(lock, "catalog", catalog)
    with pytest.raises(ValueError, match="must have one row matching the locked Zenodo source"):
        lock.fetch_from_zenodo_catalog("profiles", "catalog")
    client.assert_not_called()


def test_catalog_itself_is_verified(tmp_path, http):
    client, _ = http
    profiles = zenodo()
    catalog = catalog_bytes(profiles)
    lock = source_lock(tmp_path, {"catalog": github(data=catalog), "profiles": profiles})
    cached = seed(lock, "catalog", catalog)
    assert lock.fetch_from_zenodo_catalog("profiles", "catalog").read_bytes() == PAYLOAD
    assert set(lock.used) == {"catalog", "profiles"}
    cached.write_bytes(catalog + b"corrupt")
    with pytest.raises(ValueError, match="Checksum mismatch for catalog"):
        lock.fetch_from_zenodo_catalog("profiles", "catalog")
    assert client.call_count == 1


def test_receipts_are_bound_to_output_and_keep_manifest_history(tmp_path):
    old_pin, new_pin = github(), github(commit="b" * 40)
    data = tmp_path / "data"
    data.mkdir()
    output, manifest = data / "output.csv", data / "manifest.json"
    output.write_bytes(b"original")
    update_manifest(data, manifest, "2026-11-19", include_untracked=True)

    for pin, payload, date in (
        (old_pin, b"original", "2026-11-20"),
        (new_pin, b"changed", "2026-11-21"),
    ):
        lock = source_lock(tmp_path, {"source": pin})
        lock.fetch("source", local_path=seed(lock, "source", PAYLOAD))
        with lock.output(output, ["source"]) as staged:
            staged.write_bytes(payload)
        update_manifest(data, manifest, date, include_untracked=True)
        state = json.loads(manifest.read_text())
        assert list(state["files"]) == ["output.csv"]
        entry = state["files"]["output.csv"]
        assert entry["input_pins"] == {"source": pin}
        if pin == old_pin:
            assert state["data_version"] == "2026.11.19"
            assert entry["history"] == []
        else:
            assert state["data_version"] == "2026.11.21"
            assert entry["history"][0]["input_pins"] == {"source": old_pin}

    last_manifest = manifest.read_bytes()
    output.write_bytes(b"edited")
    with pytest.raises(ValueError, match="recorded output MD5 does not match"):
        update_manifest(data, manifest, "2026-11-22", include_untracked=True)
    assert manifest.read_bytes() == last_manifest

    receipt_path(output).unlink()
    update_manifest(data, manifest, "2026-11-22", include_untracked=True)
    entry = json.loads(manifest.read_text())["files"]["output.csv"]
    assert "input_pins" not in entry
    assert entry["history"][-1]["input_pins"] == {"source": new_pin}


def test_failed_build_preserves_previous_output_and_receipt(tmp_path):
    lock = source_lock(tmp_path, {"source": github()})
    lock.fetch("source", local_path=seed(lock, "source", PAYLOAD))
    output = tmp_path / "output.csv"
    with lock.output(output, ["source"]) as staged:
        staged.write_bytes(b"good")
    receipt = receipt_path(output).read_bytes()
    with pytest.raises(RuntimeError, match="failed"):
        with lock.output(output, ["source"]) as staged:
            staged.write_bytes(b"incomplete")
            raise RuntimeError("build failed")
    assert output.read_bytes() == b"good"
    assert receipt_path(output).read_bytes() == receipt


def test_receipt_and_output_can_be_written_to_a_new_directory(tmp_path):
    lock = source_lock(tmp_path, {"source": github()})
    lock.fetch("source", local_path=seed(lock, "source", PAYLOAD))
    output = tmp_path / "new" / "nested" / "output.csv"

    with lock.output(output, ["source"]) as staged:
        staged.write_bytes(b"new output")

    assert output.read_bytes() == b"new output"
    receipt = json.loads(receipt_path(output).read_text())
    assert receipt["output_md5"] == hashlib.md5(b"new output").hexdigest()
    assert receipt["input_pins"] == {"source": github()}
