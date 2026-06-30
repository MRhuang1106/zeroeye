import contextlib
import datetime as dt
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import verify_diagnostics


class VerifyDiagnosticsTest(unittest.TestCase):
    def make_report(self, overrides=None, module_overrides=None):
        tempdir = tempfile.TemporaryDirectory()
        root = Path(tempdir.name)
        diagnostic = root / "diagnostic"
        diagnostic.mkdir()
        commit = "abcdef12"
        logd = diagnostic / f"build-{commit}.logd"
        logd.write_bytes(b"DIAG fake encrypted archive")

        module = {
            "name": "frailbox",
            "status": "PASS",
            "elapsed_seconds": 1.25,
            "artifact": None,
            "output": "ok",
        }
        if module_overrides:
            module.update(module_overrides)

        report = {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "commit": commit,
            "diagnostic_logd": f"diagnostic/build-{commit}.logd",
            "diagnostic_logd_error": None,
            "chunked": False,
            "chunk_size_bytes": None,
            "password": "secret",
            "decrypt_command": f"encryptly unpack diagnostic/build-{commit}.logd <outdir> --password secret",
            "total_modules": 1,
            "passed": 1 if module["status"] == "PASS" else 0,
            "failed": 1 if module["status"] == "FAIL" else 0,
            "modules": [module],
            "pr_note": "Include diagnostics in the PR.",
        }
        if overrides:
            report.update(overrides)

        report_path = diagnostic / f"build-{commit}.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        self.addCleanup(tempdir.cleanup)
        return report_path

    def test_valid_report_passes(self):
        report_path = self.make_report()

        result = verify_diagnostics.validate_report(report_path)

        self.assertTrue(result.ok)
        self.assertEqual(result.passed, 1)
        self.assertEqual(result.failed, 0)
        self.assertEqual(result.errors, [])

    def test_missing_required_field_is_structural_error(self):
        report_path = self.make_report(overrides={"modules": []})
        data = json.loads(report_path.read_text(encoding="utf-8"))
        del data["generated_at"]
        report_path.write_text(json.dumps(data), encoding="utf-8")

        result = verify_diagnostics.validate_report(report_path)

        self.assertFalse(result.ok)
        self.assertIn("Missing required field: generated_at", result.errors)
        self.assertIn("Field 'total_modules' is 1, but modules contains 0 entries", result.errors)

    def test_threshold_can_fail_validation(self):
        report_path = self.make_report()

        result = verify_diagnostics.validate_report(report_path, threshold=2)

        self.assertFalse(result.ok)
        self.assertIn("Passing modules 1 is below threshold 2", result.errors)

    def test_boolean_counts_are_rejected(self):
        report_path = self.make_report()
        data = json.loads(report_path.read_text(encoding="utf-8"))
        data["total_modules"] = True
        report_path.write_text(json.dumps(data), encoding="utf-8")

        result = verify_diagnostics.validate_report(report_path)

        self.assertFalse(result.ok)
        self.assertIn("Field 'total_modules' must be int, got bool", result.errors)

    def test_json_cli_output(self):
        report_path = self.make_report()
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            exit_code = verify_diagnostics.main(["--json", str(report_path)])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["passed"], 1)
        self.assertEqual(payload["errors"], [])

    def test_run_command_reports_subprocess_failure(self):
        with mock.patch(
            "tools.verify_diagnostics.subprocess.run",
            side_effect=FileNotFoundError("missing"),
        ):
            result = verify_diagnostics.run_command(["git", "status"])

        self.assertFalse(result.ok)
        self.assertIn("executable not found", result.error)


if __name__ == "__main__":
    unittest.main()
