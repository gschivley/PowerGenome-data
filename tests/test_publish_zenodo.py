import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests

SCRIPT = Path(__file__).parents[1] / "publish_zenodo.py"
SPEC = importlib.util.spec_from_file_location("publish_zenodo", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

MANIFEST = {
    "data_version": "2026.08.14",
    "files": {
        "core_a.csv": {
            "version": "2026-08-11",
            "last_updated": "2026-08-11",
            "md5": "aaa",
        },
    },
    "sections": {
        "profiles": {
            "files": {
                "wind.parquet": {
                    "version": "2026-08-12",
                    "last_updated": "2026-08-12",
                    "md5": "bbb",
                }
            }
        },
        "existing_resource_groups": {"files": {}},
    },
}

LEGACY_STATE = {
    "metadata": {
        "title": "PowerGenome Input Data",
        "description": "old description",
        "creators": [{"name": "Schivley, Greg"}],
    },
    "zenodo_release": {
        "environment": "sandbox",
        "data_version": "2026.08.14",
        "deposition_id": "590994",
        "doi": "10.5072/zenodo.590994",
        "published": True,
        "files": {"core_a.csv": "aaa"},
    },
}

NEW_STATE = {
    "metadata": {
        "creators": [{"name": "Schivley, Greg"}],
        "sections": {
            "profiles": {"title": "PowerGenome Renewable Resource Profiles"},
        },
    },
    "releases": {
        "profiles": {
            "environment": "sandbox",
            "data_version": "2026.08.13",
            "deposition_id": "600001",
            "published": False,
            "files": {"wind.parquet": "bbb"},
        }
    },
}


class ManifestSectionTests(unittest.TestCase):
    def _write(self, root, rel_dir, content):
        d = root / rel_dir
        d.mkdir(parents=True, exist_ok=True)
        p = d / "manifest.json"
        p.write_text(json.dumps(content))
        return p

    def _args(self, manifest):
        args = mock.Mock()
        args.manifest = str(manifest)
        args.collection = None
        return args

    def test_local_manifests_are_used_per_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(
                root,
                "data",
                {"data_version": "2026.08.14", "files": {"core_a.csv": {}}},
            )
            self._write(
                root,
                "resource_profiles",
                {"data_version": "2026.08.31", "files": {"wind.parquet": {}}},
            )
            self._write(
                root,
                "existing_resource_groups",
                {"data_version": "2026.08.31", "files": {"meta.csv": {}}},
            )
            manifests = MODULE.load_manifests(
                self._args(root / "data" / "manifest.json"), root
            )
            self.assertEqual(set(manifests), set(MODULE.COLLECTIONS))
            self.assertEqual(manifests["core"]["data_version"], "2026.08.14")
            self.assertEqual(list(manifests["core"]["files"]), ["core_a.csv"])
            self.assertEqual(manifests["profiles"]["data_version"], "2026.08.31")
            self.assertEqual(list(manifests["profiles"]["files"]), ["wind.parquet"])
            self.assertEqual(
                manifests["existing_resource_groups"]["data_version"], "2026.08.31"
            )

    def test_missing_local_manifest_falls_back_to_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(
                root,
                "data",
                MANIFEST,
            )
            manifests = MODULE.load_manifests(
                self._args(root / "data" / "manifest.json"), root
            )
            # No resource_profiles/manifest.json -> falls back to sections.
            self.assertEqual(list(manifests["profiles"]["files"]), ["wind.parquet"])
            self.assertEqual(manifests["profiles"]["data_version"], "2026.08.14")
            self.assertEqual(manifests["existing_resource_groups"]["files"], {})

    def test_sections_fallback_defaults_to_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(
                root, "data", {"data_version": "2026.08.14", "files": {"a.csv": {}}}
            )
            manifests = MODULE.load_manifests(
                self._args(root / "data" / "manifest.json"), root
            )
            self.assertEqual(manifests["profiles"]["files"], {})
            self.assertEqual(manifests["existing_resource_groups"]["files"], {})
            self.assertEqual(manifests["profiles"]["data_version"], "2026.08.14")

    def test_per_collection_data_version_is_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(
                root,
                "data",
                {"data_version": "2026.08.14", "files": {"core_a.csv": {}}},
            )
            self._write(
                root,
                "resource_profiles",
                {"data_version": "2026.08.31", "files": {"wind.parquet": {}}},
            )
            manifests = MODULE.load_manifests(
                self._args(root / "data" / "manifest.json"), root
            )
            # Updating profiles must NOT touch the core version.
            self.assertEqual(manifests["core"]["data_version"], "2026.08.14")
            self.assertEqual(manifests["profiles"]["data_version"], "2026.08.31")

    def test_unknown_section_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "data_version": "2026.08.14",
                        "files": {},
                        "sections": {"bogus": {"files": {}}},
                    }
                )
            )
            with self.assertRaises(SystemExit):
                MODULE.load_manifest(path)


class StateMigrationTests(unittest.TestCase):
    def test_legacy_zenodo_release_migrates_to_core(self):
        released = MODULE.section_release(LEGACY_STATE, "core")
        self.assertEqual(released["deposition_id"], "590994")
        self.assertEqual(released["doi"], "10.5072/zenodo.590994")
        self.assertTrue(released["published"])

    def test_legacy_state_has_no_non_core_sections(self):
        self.assertEqual(MODULE.section_release(LEGACY_STATE, "profiles"), {})

    def test_new_releases_state_wins_over_legacy(self):
        state = dict(LEGACY_STATE)
        state["releases"] = {"core": {"deposition_id": "700002", "published": False}}
        released = MODULE.section_release(state, "core")
        self.assertEqual(released["deposition_id"], "700002")

    def test_new_state_section_lookup(self):
        released = MODULE.section_release(NEW_STATE, "profiles")
        self.assertEqual(released["deposition_id"], "600001")
        self.assertEqual(MODULE.section_release(NEW_STATE, "core"), {})


class MetadataTests(unittest.TestCase):
    def test_section_metadata_merges_overrides(self):
        meta = MODULE.section_metadata(NEW_STATE, "profiles")
        self.assertEqual(meta["title"], "PowerGenome Renewable Resource Profiles")
        self.assertEqual(meta["creators"], [{"name": "Schivley, Greg"}])
        self.assertNotIn("sections", meta)

    def test_section_metadata_defaults_without_overrides(self):
        meta = MODULE.section_metadata(LEGACY_STATE, "existing_resource_groups")
        self.assertEqual(meta["creators"], [{"name": "Schivley, Greg"}])
        self.assertNotIn("title", meta)  # per-release field, not shared


class DescriptionTests(unittest.TestCase):
    def test_description_uses_section_title_and_files(self):
        files = MANIFEST["sections"]["profiles"]["files"]
        description = MODULE.build_description(
            MANIFEST, "profiles", files, ["wind.parquet"], [], [], True
        )
        self.assertIn("PowerGenome Renewable Resource Profiles", description)
        self.assertIn("wind.parquet", description)
        self.assertIn("2026.08.14", description)
        self.assertNotIn("core_a.csv", description)

    def test_description_lists_changes(self):
        files = MANIFEST["files"]
        description = MODULE.build_description(
            MANIFEST, "core", files, [], ["core_a.csv"], ["gone.csv"], False
        )
        self.assertIn("Files updated in this release", description)
        self.assertIn("Files removed in this release", description)
        self.assertIn("gone.csv", description)

    def test_readme_body_sits_below_change_note(self):
        import tempfile as _tempfile

        files = MANIFEST["files"]
        with _tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "README.md").write_text(
                "# Collection docs\n\nA table:\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n"
            )
            readme_html = MODULE.readme_to_html(data_dir)
            description = MODULE.build_description(
                MANIFEST, "core", files, ["core_a.csv"], [], [], False, readme_html
            )
        body = "<h1>Collection docs</h1>" if MODULE.markdown else "<pre>"
        self.assertIn(body, description)
        # README body appears after the change note ("Files added...") ...
        self.assertLess(
            description.index("Files added in this release"),
            description.index("Collection docs"),
        )
        # ... and before the licensing/per-file sections.
        self.assertLess(
            description.index("Collection docs"),
            description.index("Licensing"),
        )

    def test_readme_to_html_empty_without_readme(self):
        import tempfile as _tempfile

        with _tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(MODULE.readme_to_html(Path(tmp)), "")

    def test_release_description_prepends_custom_section_prose(self):
        files = MANIFEST["sections"]["profiles"]["files"]
        base = {
            "description": "<p><strong>Site mapping.</strong> CPA_ID, Site, profile_dist.</p>"
        }
        description = MODULE.build_release_description(
            MANIFEST, "profiles", files, ["wind.parquet"], [], [], True, base
        )
        self.assertTrue(description.startswith("<p><strong>Site mapping.</strong>"))
        self.assertIn("PowerGenome Renewable Resource Profiles", description)

    def test_release_description_without_override_is_unpadded(self):
        files = MANIFEST["files"]
        description = MODULE.build_release_description(
            MANIFEST, "core", files, [], [], [], True, {}
        )
        self.assertTrue(description.startswith("<p>PowerGenome Input Data"))


class ManifestEntryTests(unittest.TestCase):
    def test_real_profiles_manifest_has_six_files(self):
        import json as _json

        real = _json.loads(
            (
                Path(__file__).parents[1] / "resource_profiles" / "manifest.json"
            ).read_text()
        )
        self.assertIn("data_version", real)
        profiles = real["files"]
        self.assertEqual(len(profiles), 6)
        self.assertIn("solar_site_mapping_20240801.parquet", profiles)
        for name, entry in profiles.items():
            self.assertIn("md5", entry)
            self.assertIn("source", entry["sources"][0])
            self.assertEqual(entry["license"], "cc-by-4.0")

    def test_real_existing_resource_groups_manifest_has_fourteen_files(self):
        import json as _json

        real = _json.loads(
            (
                Path(__file__).parents[1] / "existing_resource_groups" / "manifest.json"
            ).read_text()
        )
        self.assertIn("data_version", real)
        self.assertEqual(len(real["files"]), 14)
        self.assertNotIn("existing_osw_profiles.csv", real["files"])
        for name, entry in real["files"].items():
            self.assertIn("md5", entry)
            self.assertIn("source", entry["sources"][0])


def _mkargs(manifest_path, collection=None):
    args = mock.Mock()
    args.manifest = str(manifest_path)
    args.collection = collection
    return args


class DryRunTests(unittest.TestCase):
    def test_dry_run_covers_all_sections(self):
        import io
        from contextlib import redirect_stdout

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "data" / "core_a.csv").write_text("a")
            manifest = {
                "data_version": "2026.08.14",
                "files": {"core_a.csv": {"version": "2026-08-11", "md5": "x"}},
            }
            manifest_path = root / "data" / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            with mock.patch.object(MODULE.Path, "cwd", return_value=root):
                out = io.StringIO()
                with redirect_stdout(out):
                    MODULE.dry_run(_mkargs(manifest_path))
            text = out.getvalue()
        self.assertIn("[core] PowerGenome Input Data", text)
        self.assertIn("ok      core_a.csv", text)
        self.assertIn("[profiles] PowerGenome Renewable Resource Profiles", text)
        self.assertIn("deposit skipped", text)
        self.assertIn(
            "[existing_resource_groups] PowerGenome Existing Renewable Resource Groups",
            text,
        )


class CollectionFilterTests(unittest.TestCase):
    def test_requested_sections_defaults_to_all(self):
        args = mock.Mock(collection=None)
        self.assertEqual(MODULE.requested_sections(args), list(MODULE.COLLECTIONS))

    def test_requested_sections_honors_repeated_flags(self):
        args = mock.Mock(collection=["profiles", "core"])
        self.assertEqual(MODULE.requested_sections(args), ["profiles", "core"])

    def test_dry_run_filters_to_requested_collection(self):
        import io
        from contextlib import redirect_stdout

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "resource_profiles").mkdir()
            (root / "resource_profiles" / "wind.parquet").write_text("w")
            manifest = {
                "data_version": "2026.08.14",
                "files": {"core_a.csv": {"version": "2026-08-11", "md5": "x"}},
            }
            profiles_manifest = {
                "data_version": "2026.08.31",
                "files": {"wind.parquet": {"version": "2026-08-12", "md5": "y"}},
            }
            (root / "data" / "manifest.json").write_text(json.dumps(manifest))
            (root / "resource_profiles" / "manifest.json").write_text(
                json.dumps(profiles_manifest)
            )
            with mock.patch.object(MODULE.Path, "cwd", return_value=root):
                out = io.StringIO()
                with redirect_stdout(out):
                    MODULE.dry_run(
                        _mkargs(root / "data" / "manifest.json", ["profiles"])
                    )
            text = out.getvalue()
        self.assertNotIn("[core]", text)
        self.assertIn("[profiles] PowerGenome Renewable Resource Profiles", text)
        self.assertIn("[data version 2026.08.31]", text)
        self.assertIn("ok      wind.parquet", text)
        self.assertNotIn("existing_resource_groups", text)

    def test_unknown_collection_rejected_by_parser(self):
        parser = MODULE.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--collection", "bogus"])


class DepositionOverrideTests(unittest.TestCase):
    def test_parser_exposes_deposition_id(self):
        parser = MODULE.build_parser()
        args = parser.parse_args(["--deposition-id", "22233228"])
        self.assertEqual(args.deposition_id, "22233228")

    def test_deposition_id_overrides_saved_state(self):
        """--deposition-id must resume the given draft even when saved state
        points at a different (or published) deposition."""
        args = mock.Mock()
        args.deposition_id = "22233228"
        args.allow_dirty = True
        args.publish = False
        args.sleep_seconds = 0
        args.upload_retries = 0
        args.upload_retry_delay = 0

        session = mock.Mock()
        # get_deposition for the resumed draft
        draft = {
            "id": 22233228,
            "conceptrecid": 111,
            "links": {
                "bucket": "https://bucket.example",
                "files": "https://files.example",
            },
        }
        draft_response = self._response(200)
        draft_response.json.return_value = draft
        records_response = self._response(200)
        records_response.json.return_value = {"hits": {"hits": []}}
        files_response = self._response(200)
        files_response.json.return_value = []

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "core_a.csv").write_text("a")
            upload_response = self._response(200)
            upload_response.json.return_value = {
                "checksum": f"md5:{MODULE.md5_for_file(data_dir / 'core_a.csv')}"
            }
            metadata_response = self._response(200)
            metadata_response.json.return_value = {"metadata": {}}
            session.request.side_effect = [
                draft_response,
                records_response,
                files_response,
                upload_response,
                metadata_response,
            ]
            manifest = {
                "data_version": "2026.08.14",
                "files": {
                    "core_a.csv": {
                        "version": "2026-08-11",
                        "md5": MODULE.md5_for_file(data_dir / "core_a.csv"),
                    }
                },
            }
            manifest_path = root / "data" / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            state = {
                "releases": {
                    "core": {
                        "environment": "production",
                        "deposition_id": "999999",
                        "published": True,
                    }
                }
            }
            with mock.patch.object(MODULE.time, "sleep"):
                result = MODULE.release_section(
                    args,
                    session,
                    "https://zenodo.org/api",
                    "production",
                    manifest,
                    "core",
                    manifest["files"],
                    data_dir,
                    state,
                )
        self.assertEqual(result["summary"]["deposition_id"], "22233228")
        # The resumed draft must be fetched (not a new version created), then
        # records search, draft files, upload, and metadata update follow.
        self.assertEqual(session.request.call_count, 5)

    def _response(self, status=200, checksum="md5:abc"):
        response = mock.Mock()
        response.status_code = status
        response.raise_for_status.side_effect = (
            None if status < 400 else requests.HTTPError(f"HTTP {status}")
        )
        response.json.return_value = {"checksum": checksum}
        return response


class UploadRetryTests(unittest.TestCase):
    def _session(self, responses):
        """Session whose request() returns responses in order, then raises."""
        session = mock.Mock()
        session.request.side_effect = responses
        return session

    def _response(self, status=200, checksum="md5:abc"):
        response = mock.Mock()
        response.status_code = status
        response.raise_for_status.side_effect = (
            None if status < 400 else requests.HTTPError(f"HTTP {status}")
        )
        response.json.return_value = {"checksum": checksum}
        return response

    def _file(self, tmp):
        p = Path(tmp) / "data.parquet"
        p.write_bytes(b"data")
        return p

    def _ok_response(self, path):
        return self._response(checksum=f"md5:{MODULE.md5_for_file(path)}")

    def test_retries_on_connection_error_then_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._file(tmp)
            expected_md5 = MODULE.md5_for_file(path)
            session = self._session(
                [
                    requests.exceptions.ConnectionError("EOF occurred"),
                    self._ok_response(path),
                ]
            )
            with mock.patch.object(MODULE.time, "sleep") as sleep:
                result = MODULE.upload_file(session, "https://bucket.example", path)
        self.assertEqual(session.request.call_count, 2)
        self.assertEqual(result["local_md5"], expected_md5)
        sleep.assert_called_once_with(10.0)

    def test_retries_on_http_500_then_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._file(tmp)
            expected_md5 = MODULE.md5_for_file(path)
            session = self._session([self._response(500), self._ok_response(path)])
            with mock.patch.object(MODULE.time, "sleep"):
                result = MODULE.upload_file(session, "https://bucket.example", path)
        self.assertEqual(session.request.call_count, 2)
        self.assertEqual(result["local_md5"], expected_md5)

    def test_exhausts_retries_and_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(
                [requests.exceptions.ConnectionError("EOF occurred")] * 4
            )
            with mock.patch.object(MODULE.time, "sleep"):
                with self.assertRaises(requests.exceptions.ConnectionError):
                    MODULE.upload_file(
                        session, "https://bucket.example", self._file(tmp)
                    )
        self.assertEqual(session.request.call_count, 4)

    def test_retry_delay_doubles(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._file(tmp)
            session = self._session(
                [
                    requests.exceptions.ConnectionError("x"),
                    requests.exceptions.ConnectionError("x"),
                    self._ok_response(path),
                ]
            )
            with mock.patch.object(MODULE.time, "sleep") as sleep:
                MODULE.upload_file(
                    session,
                    "https://bucket.example",
                    path,
                    retries=3,
                    retry_delay=2.0,
                )
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [4.0, 8.0])

    def test_zero_byte_upload_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "empty.parquet"
            p.write_bytes(b"")
            session = self._session([])
            with self.assertRaises(ValueError):
                MODULE.upload_file(session, "https://bucket.example", p)
        self.assertEqual(session.request.call_count, 0)

    def test_parser_exposes_retry_flags(self):
        parser = MODULE.build_parser()
        args = parser.parse_args(
            ["--upload-retries", "5", "--upload-retry-delay", "10"]
        )
        self.assertEqual(args.upload_retries, 5)
        self.assertEqual(args.upload_retry_delay, 10.0)


class RetryRequestTests(unittest.TestCase):
    def test_retryable_status_codes_are_retried(self):
        for status in MODULE.RETRYABLE_STATUS_CODES:
            with self.subTest(status=status):
                session = mock.Mock()
                session.request.side_effect = [
                    self._response(status),
                    self._response(200),
                ]
                with mock.patch.object(MODULE.time, "sleep"):
                    response = MODULE.retry_request(
                        session, "GET", "https://example.com/api"
                    )
                self.assertEqual(session.request.call_count, 2)
                self.assertEqual(response.status_code, 200)

    def test_non_retryable_status_is_not_retried(self):
        session = mock.Mock()
        session.request.side_effect = [self._response(400)]
        with mock.patch.object(MODULE.time, "sleep"):
            response = MODULE.retry_request(session, "GET", "https://example.com/api")
        self.assertEqual(session.request.call_count, 1)
        self.assertEqual(response.status_code, 400)

    def test_data_factory_reopens_stream_per_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.bin"
            path.write_bytes(b"data")
            session = mock.Mock()
            session.request.side_effect = [
                requests.exceptions.ConnectionError("EOF"),
                self._response(200),
            ]
            with mock.patch.object(MODULE.time, "sleep"):
                MODULE.retry_request(
                    session,
                    "PUT",
                    "https://example.com/bucket/data.bin",
                    data_factory=lambda: path.open("rb"),
                )
        # Each attempt must open a fresh handle (the first was consumed).
        self.assertEqual(session.request.call_count, 2)

    def test_publish_404_checks_deposition_state(self):
        session = mock.Mock()
        not_found = self._response(404)
        submitted = self._response(200, checksum="md5:abc")
        submitted.json.return_value = {"submitted": True}
        session.request.side_effect = [not_found, submitted]
        with mock.patch.object(MODULE.time, "sleep"):
            result = MODULE.publish_deposition(
                session, "https://zenodo.org/api", "12345"
            )
        self.assertEqual(result["submitted"], True)

    def test_publish_404_when_not_submitted_raises(self):
        session = mock.Mock()
        session.request.side_effect = [self._response(404), self._response(404)]
        with mock.patch.object(MODULE.time, "sleep"):
            with self.assertRaises(SystemExit):
                MODULE.publish_deposition(session, "https://zenodo.org/api", "12345")

    def _response(self, status=200, checksum="md5:abc"):
        response = mock.Mock()
        response.status_code = status
        response.raise_for_status.side_effect = (
            None if status < 400 else requests.HTTPError(f"HTTP {status}")
        )
        response.json.return_value = {"checksum": checksum}
        return response


if __name__ == "__main__":
    unittest.main()
