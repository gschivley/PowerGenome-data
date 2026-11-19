"""Resolve immutable build inputs and bind their provenance to output bytes."""

import csv
import hashlib
import json
import re
from contextlib import contextmanager
from pathlib import Path
from stat import S_IMODE
from tempfile import NamedTemporaryFile
from urllib.parse import quote, urlsplit

import requests

DEFAULT_LOCK = Path(__file__).with_name("sources.lock.json")
DEFAULT_CACHE = Path("cache_data/reeds-sources")
RECEIPT_SUFFIX = ".sources.json"
ZENODO_TEMPLATE = "https://zenodo.org/records/{record_id}/files/{filename}"


def add_source_options(parser, cache_dir=DEFAULT_CACHE):
    parser.add_argument("--cache-dir", type=Path, default=cache_dir)
    parser.add_argument(
        "--sources-lock", type=Path, default=DEFAULT_LOCK,
        help="Input lockfile (default: sources.lock.json beside the builders)",
    )


def digest(path, algorithm):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def _safe_parts(path):
    parts = path.split("/")
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", part) or part in {".", ".."} for part in parts):
        raise ValueError(f"Invalid locked path: {path!r}")
    return parts


def _source_url(pin):
    kind = pin["kind"]
    checksum = pin["checksum"]
    algorithm = "md5" if kind == "zenodo" else "sha256"
    length = 32 if algorithm == "md5" else 64
    if not re.fullmatch(rf"{algorithm}:[0-9a-f]{{{length}}}", checksum):
        raise ValueError(f"Invalid {algorithm} checksum: {checksum!r}")

    if kind == "github":
        repo, commit, path = (pin[key] for key in ("repository", "commit", "path"))
        if len(_safe_parts(repo)) != 2 or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("GitHub pins require an owner/repository and a full commit SHA")
        _safe_parts(path)
        lfs = pin.get("lfs", False)
        if not isinstance(lfs, bool):
            raise ValueError("GitHub lfs must be a boolean")
        if lfs:
            return f"https://media.githubusercontent.com/media/{repo}/{commit}/{path}"
        return f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"

    if kind == "zenodo":
        record_id, filename = pin["record_id"], pin["filename"]
        if type(record_id) is not int or record_id <= 0 or len(_safe_parts(filename)) != 1:
            raise ValueError("Zenodo pins require a numeric version record ID and a filename")
        return ZENODO_TEMPLATE.format(record_id=record_id, filename=quote(filename))

    if kind == "https":
        url = pin["url"]
        parts = urlsplit(url)
        if (parts.scheme != "https" or not re.fullmatch(r"[a-z0-9.-]+", parts.netloc)
                or parts.query or parts.fragment):
            raise ValueError(f"Invalid pinned HTTPS URL: {url!r}")
        if parts.netloc in {
            "github.com", "raw.githubusercontent.com", "media.githubusercontent.com", "zenodo.org",
        }:
            raise ValueError("GitHub and Zenodo URLs require provider-specific source pins")
        _safe_parts(parts.path.removeprefix("/"))
        if {"main", "latest"} & set(parts.path.lower().split("/")):
            raise ValueError(f"Moving URL is invalid for a source pin: {url!r}")
        return url

    raise ValueError(f"Unsupported source kind: {kind!r}")


def receipt_path(output):
    return Path(f"{output}{RECEIPT_SUFFIX}")


def read_output_pins(output, checksum):
    """Read a build's verified pins only if its receipt matches the output bytes."""
    path = receipt_path(output)
    if not path.exists():
        return None
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt["output_md5"] != checksum:
            raise ValueError(f"recorded output MD5 does not match {checksum}")
        pins = receipt["input_pins"]
        if not isinstance(pins, dict) or not pins:
            raise ValueError("expected a nonempty pin snapshot")
        for pin in pins.values():
            _source_url(pin)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid source receipt {path}: {exc}") from exc
    return pins


class SourceLock:
    def __init__(self, cache_dir=DEFAULT_CACHE, lock_path=DEFAULT_LOCK):
        self.cache_dir = Path(cache_dir)
        document = json.loads(Path(lock_path).read_text(encoding="utf-8"))
        if (
            not isinstance(document, dict)
            or document.get("lock_version") != 1
            or not isinstance(document.get("sources"), dict)
        ):
            raise ValueError(f"Invalid source lockfile: {lock_path}")
        self.sources = document["sources"]
        self.used = {}

    def _resolve(self, source_id):
        try:
            pin = self.sources[source_id]
        except KeyError:
            raise ValueError(f"Missing source pin: {source_id}") from None
        try:
            return pin, _source_url(pin)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid source pin {source_id}: {exc}") from exc

    def cache_path(self, source_id):
        pin, url = self._resolve(source_id)
        if pin["kind"] == "github":
            return self.cache_dir / "github" / pin["repository"] / pin["commit"] / pin["path"]
        parts = urlsplit(url)
        return self.cache_dir / parts.netloc / parts.path.lstrip("/")

    def fetch(self, source_id, *, local_path=None):
        pin, url = self._resolve(source_id)
        path = self.cache_path(source_id) if local_path is None else Path(local_path)
        algorithm, expected = pin["checksum"].split(":")
        if local_path is not None or path.exists():
            self._verify(path, source_id, algorithm, expected)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
                temporary = Path(temp.name)
                try:
                    with requests.get(url, stream=True, timeout=(15, 90)) as response:
                        response.raise_for_status()
                        for chunk in response.iter_content(1024 * 1024):
                            temp.write(chunk)
                    temp.flush()
                    self._verify(temporary, source_id, algorithm, expected)
                    temporary.replace(path)
                except requests.RequestException as exc:
                    raise RuntimeError(f"Could not download {source_id} from {url}: {exc}") from exc
                finally:
                    temporary.unlink(missing_ok=True)
        self.used[source_id] = pin
        return path

    @staticmethod
    def _verify(path, source_id, algorithm, expected):
        actual = digest(path, algorithm)
        if actual != expected:
            raise ValueError(
                f"Checksum mismatch for {source_id} at {path}: "
                f"expected {algorithm}:{expected}, got {algorithm}:{actual}"
            )

    def fetch_from_zenodo_catalog(self, source_id, catalog_id):
        pin, _ = self._resolve(source_id)
        if pin["kind"] != "zenodo":
            raise ValueError(f"{source_id} must have a Zenodo pin")
        catalog = self.fetch(catalog_id)
        with catalog.open(newline="", encoding="utf-8-sig") as stream:
            rows = [row for row in csv.DictReader(stream) if row.get("filename") == pin["filename"]]
        expected = (str(pin["record_id"]), pin["checksum"].removeprefix("md5:"), ZENODO_TEMPLATE)
        if len(rows) != 1 or tuple(rows[0].get(key) for key in ("record_id", "md5", "url_base")) != expected:
            raise ValueError(f"{catalog_id} must have one row matching the locked Zenodo source {source_id}")
        return self.fetch(source_id)

    @contextmanager
    def output(self, output, source_ids):
        """Stage the output and record verified pins for the bytes this build wrote."""
        output = Path(output)
        pins = {source_id: self.used[source_id] for source_id in source_ids}
        with NamedTemporaryFile(dir=output.parent, suffix=output.suffix, delete=False) as temp:
            staged_output = Path(temp.name)
        try:
            yield staged_output
            checksum = digest(staged_output, "md5")
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=output.parent, delete=False,
            ) as temp:
                staged_receipt = Path(temp.name)
                try:
                    json.dump({"output_md5": checksum, "input_pins": pins}, temp, indent=2, sort_keys=True)
                    temp.write("\n")
                    temp.flush()
                    staged_output.chmod(S_IMODE(output.stat().st_mode) if output.exists() else 0o644)
                    staged_output.replace(output)
                    staged_receipt.replace(receipt_path(output))
                finally:
                    staged_receipt.unlink(missing_ok=True)
        finally:
            staged_output.unlink(missing_ok=True)
