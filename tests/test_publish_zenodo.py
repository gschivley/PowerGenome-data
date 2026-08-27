import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "publish_zenodo.py"
SPEC = importlib.util.spec_from_file_location("publish_zenodo", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def make_manifest():
    return {
        "data_version": "2026.08.14",
        "files": {
            "alpha.csv": {
                "version": "2026-08-14",
                "last_updated": "2026-08-14",
                "md5": "a1",
                "license": "public-domain",
                "sources": [
                    {"source": "FRED series", "source_url": "https://fred.example"}
                ],
                "history": [
                    {
                        "version": "2026-08-01",
                        "last_updated": "2026-08-01",
                        "md5": "a0",
                    }
                ],
            },
            "beta.parquet": {
                "version": "2026-08-11",
                "last_updated": "2026-08-11",
                "md5": "b1",
                "license": "cc-by-4.0",
                "sources": [{"source": "ReEDS"}],
                "history": [],
            },
            "gamma.csv": {
                "version": "2026-08-11",
                "last_updated": "2026-08-12",
                "md5": "g1",
                "license": "cc-zero",
                "sources": [{"source": "EIA"}],
                "history": [],
            },
        },
    }


class DescriptionTests(unittest.TestCase):
    def _build(self, **kwargs):
        defaults = dict(
            added=[], updated=[], removed=[], removed_details={}, initial=False
        )
        defaults.update(kwargs)
        return MODULE.build_description(make_manifest(), **defaults)

    def test_updated_shows_version_transition(self):
        desc = self._build(updated=["alpha.csv"])
        self.assertIn("(2026-08-01 &rarr; 2026-08-14)", desc)

    def test_updated_without_history_has_no_transition(self):
        desc = self._build(updated=["beta.parquet"])
        self.assertIn("<code>beta.parquet</code>", desc)
        self.assertNotIn("beta.parquet</code> (", desc)

    def test_removed_files_render_prior_details_and_note(self):
        prior = {
            "version": "2026-08-01",
            "last_updated": "2026-08-01",
            "md5": "x0",
            "license": "cc-by-4.0",
            "sources": [{"source": "NERC LTRA"}],
        }
        desc = self._build(
            removed=["delta.csv"],
            removed_details={"delta.csv": prior},
            removal_notes={"delta.csv": "superseded by nerc_reserve_margins.csv"},
        )
        self.assertIn("Files removed in this release", desc)
        self.assertIn("<code>delta.csv</code>", desc)
        self.assertIn("2026-08-01", desc)
        self.assertIn("superseded by nerc_reserve_margins.csv", desc)
        self.assertIn("NERC LTRA", desc)

    def test_removed_without_prior_state_renders_unknowns(self):
        desc = self._build(removed=["delta.csv"], removed_details={})
        self.assertIn("delta.csv", desc)
        self.assertIn("Unknown", desc)

    def test_initial_release_note(self):
        desc = self._build(initial=True)
        self.assertIn("initial release", desc)
        self.assertIn("All 3 files are new", desc)

    def test_no_changes_note(self):
        desc = self._build()
        self.assertIn("No file changes in this release", desc)

    def test_licensing_paragraph_derived_from_files(self):
        desc = self._build()
        self.assertIn("CC0, public domain dedication", desc)
        self.assertIn("Creative Commons Attribution 4.0 International (CC BY 4.0)", desc)
        self.assertIn("Public domain (U.S. government)", desc)
        # Files are listed under their license group.
        self.assertIn("<code>alpha.csv</code>", desc)
        self.assertIn("<code>beta.parquet</code>", desc)
        self.assertIn("<code>gamma.csv</code>", desc)
        # Old hardcoded prose about specific sources is gone.
        self.assertNotIn("NERC LTRA data is", desc)

    def test_last_updated_row_hidden_when_equal_to_version(self):
        alpha = MODULE.describe_file("alpha.csv", make_manifest()["files"]["alpha.csv"])
        self.assertNotIn("Last updated", alpha)
        # gamma.csv diverges (version 2026-08-11, last_updated 2026-08-12).
        gamma = MODULE.describe_file("gamma.csv", make_manifest()["files"]["gamma.csv"])
        self.assertIn("Last updated", gamma)
        self.assertIn("2026-08-12", gamma)

    def test_published_at_and_git_sha_in_intro(self):
        desc = self._build(published_at="2026-08-27", git_sha="8f3a2c1")
        self.assertIn("published 2026-08-27", desc)
        self.assertIn("<code>8f3a2c1</code>", desc)

    def test_no_provenance_when_omitted(self):
        desc = self._build()
        self.assertNotIn("published ", desc)
        self.assertNotIn("git commit", desc)


class MetadataHelperTests(unittest.TestCase):
    def test_meta_hash_stable_and_sensitive(self):
        m = make_manifest()
        h1 = MODULE.manifest_meta_hash(m)
        h2 = MODULE.manifest_meta_hash(m)
        self.assertEqual(h1, h2)
        # generated_at_utc is ignored.
        m["generated_at_utc"] = "irrelevant"
        self.assertEqual(h1, MODULE.manifest_meta_hash(m))
        # A source-text edit changes the hash.
        m["files"]["alpha.csv"]["sources"][0]["source"] = "changed text"
        self.assertNotEqual(h1, MODULE.manifest_meta_hash(m))

    def test_undocumented_files(self):
        m = make_manifest()
        self.assertEqual(MODULE.undocumented_files(m["files"]), [])
        m["files"]["alpha.csv"].pop("license")
        m["files"]["beta.parquet"]["sources"] = [{"source": "Unknown - document me"}]
        self.assertEqual(
            MODULE.undocumented_files(m["files"]), ["alpha.csv", "beta.parquet"]
        )

    def test_normalize_released_files_legacy_and_new(self):
        raw = {
            "a.csv": "md5a",
            "b.csv": {"md5": "md5b", "license": "cc-zero", "version": "2026-08-01"},
        }
        norm = MODULE.normalize_released_files(raw)
        self.assertEqual(norm["a.csv"]["md5"], "md5a")
        self.assertEqual(norm["b.csv"]["license"], "cc-zero")
        self.assertEqual(MODULE.normalize_released_files({}), {})


if __name__ == "__main__":
    unittest.main()
