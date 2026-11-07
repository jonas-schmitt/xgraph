"""Installed CPU onboarding: local synthetic data, no model or network."""

import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_EXAMPLE = (
    "import runpy, sys; from pathlib import Path; "
    "script = Path(sys.argv[1]).resolve(); "
    "sys.path.insert(0, str(script.parent)); "
    "sys.argv = sys.argv[1:]; "
    "runpy.run_path(str(script), run_name='__main__')"
)


class SourceSmoke(unittest.TestCase):
    def run_example(self, name, *, output=False):
        with tempfile.TemporaryDirectory(prefix="xgraph-smoke-") as temporary:
            # Only examples/ joins sys.path, for the examples' toy_kg helper.
            # xgraph must still resolve through the installed environment.
            arguments = [
                sys.executable,
                "-I",
                "-c",
                RUN_EXAMPLE,
                str(ROOT / "examples" / name),
            ]
            if output:
                arguments += ["--out", str(Path(temporary) / "output")]
            result = subprocess.run(
                arguments,
                cwd=temporary,
                env={
                    **{
                        key: value
                        for key, value in os.environ.items()
                        if not key.startswith(("XGRAPH_", "XG_"))
                    },
                    "CUDA_VISIBLE_DEVICES": "",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                },
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return result.stdout

    def test_readme_quick_start(self):
        self.assertEqual(
            ast.literal_eval(self.run_example("quick_start.py").strip()),
            (("atlas", "creator", "ada"),),
        )

    def test_toy_graph_resolves_ids_and_binds_answer(self):
        report = json.loads(self.run_example("toy_kg.py", output=True))
        self.assertEqual(report["entity"], ["item:atlas"])
        self.assertEqual(report["fact"], [["item:atlas", "rel:creator", "person:ada"]])
        self.assertEqual(report["path"], ["place:cedar"])
        self.assertEqual(report["answer"], ["place:cedar"])
        self.assertEqual(report["anchored_path"], ["place:cedar"])
        self.assertFalse(report["escape_resolved"])
        self.assertTrue(report["rejected_wrong_answer"])

    def test_numeric_versions_keep_base_reader(self):
        report = json.loads(self.run_example("kg_versions.py", output=True))
        self.assertEqual(report["base"], ["Q3"])
        self.assertEqual(report["base_after_edit"], ["Q3"])
        self.assertEqual(report["edited"], ["Q4"])
        self.assertEqual(report["loaded_edits"], {"updated": [["Q2", "P2", "Q4"]]})
        self.assertTrue(report["rejected_old_answer"])

    def test_application_versions_keep_base_and_delete_pair(self):
        report = json.loads(self.run_example("string_versions.py", output=True))
        self.assertEqual(report["base"], ["place:cedar"])
        self.assertEqual(report["base_again"], ["place:cedar"])
        self.assertEqual(report["updated"], ["place:delta"])
        self.assertEqual(report["deleted"], [])
        self.assertEqual(
            report["loaded_edits"],
            {
                "updated": [["person:ada", "rel:born-in", "place:delta"]],
                "deleted": [["person:ada", "rel:born-in", None]],
            },
        )


if __name__ == "__main__":
    unittest.main()
