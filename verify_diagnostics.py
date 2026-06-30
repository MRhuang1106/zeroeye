#!/usr/bin/env python3
"""
Validate build diagnostic metadata emitted by build.py.

The PR workflow validates diagnostic bundles after they are pushed. This tool
lets contributors catch malformed diagnostic JSON, missing .logd references,
and module pass-count threshold failures before opening a PR.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


COMMIT_RE = re.compile(r"^[0-9a-f]{8}$")
LOGD_RE = re.compile(r"^diagnostic/build-(?P<commit>[0-9a-f]{8})(?:-part(?P<part>\d{3}))?\.logd$")
REQUIRED_FIELDS: dict[str, tuple[type, ...]] = {
    "generated_at": (str,),
    "commit": (str,),
    "diagnostic_logd": (str, list, type(None)),
    "diagnostic_logd_error": (str, type(None)),
    "chunked": (bool,),
    "chunk_size_bytes": (int, type(None)),
    "password": (str, type(None)),
    "decrypt_command": (str, type(None)),
    "total_modules": (int,),
    "passed": (int,),
    "failed": (int,),
    "modules": (list,),
    "pr_note": (str,),
}

MODULE_FIELDS: dict[str, tuple[type, ...]] = {
    "name": (str,),
    "status": (str,),
    "elapsed_seconds": (int, float),
    "artifact": (str, type(None)),
    "output": (str,),
}


@dataclass
class CommandResult:
    ok: bool
    command: list[str]
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


@dataclass
class VerificationResult:
    path: str
    ok: bool
    threshold: int
    total_modules: int | None = None
    passed: int | None = None
    failed: int | None = None
    errors: list[str] | None = None
    warnings: list[str] | None = None
    modules: list[dict[str, Any]] | None = None

    def to_dict(self, verbose: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "path": self.path,
            "threshold": self.threshold,
            "total_modules": self.total_modules,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors or [],
            "warnings": self.warnings or [],
        }
        if verbose:
            payload["modules"] = self.modules or []
        return payload


def run_command(args: Sequence[str], cwd: Path | None = None, timeout: int = 10) -> CommandResult:
    """Run an external command and return a structured error instead of raising."""
    command = list(args)
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        return CommandResult(
            ok=False,
            command=command,
            returncode=None,
            error=f"Could not run {' '.join(command)!r}: executable not found ({exc}).",
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return CommandResult(
            ok=False,
            command=command,
            returncode=None,
            stdout=stdout,
            stderr=stderr,
            error=f"Command {' '.join(command)!r} timed out after {timeout}s.",
        )
    except OSError as exc:
        return CommandResult(
            ok=False,
            command=command,
            returncode=None,
            error=f"Could not run {' '.join(command)!r}: {exc}.",
        )

    stderr = completed.stderr.strip()
    stdout = completed.stdout.strip()
    if completed.returncode != 0:
        detail = stderr or stdout or "no output"
        return CommandResult(
            ok=False,
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            error=f"Command {' '.join(command)!r} exited {completed.returncode}: {detail}",
        )

    return CommandResult(
        ok=True,
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def find_repo_root(start: Path) -> tuple[Path, str | None]:
    """Find a repo root with git when available, then fall back to parent search."""
    start_dir = start if start.is_dir() else start.parent
    git_result = run_command(["git", "rev-parse", "--show-toplevel"], cwd=start_dir, timeout=5)
    if git_result.ok and git_result.stdout.strip():
        return Path(git_result.stdout.strip()).resolve(), None

    for candidate in (start_dir.resolve(), *start_dir.resolve().parents):
        if (candidate / ".git").exists():
            return candidate, git_result.error

    if start_dir.name == "diagnostic":
        return start_dir.parent.resolve(), git_result.error
    return start_dir.resolve(), git_result.error


def latest_diagnostic_report(repo_root: Path) -> Path | None:
    diagnostic_dir = repo_root / "diagnostic"
    if not diagnostic_dir.exists():
        return None
    reports = sorted(
        diagnostic_dir.glob("build-*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return reports[0] if reports else None


def read_json_report(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(f"Diagnostic JSON not found: {path}")
    except json.JSONDecodeError as exc:
        errors.append(f"Diagnostic JSON is not valid JSON: {exc.msg} at line {exc.lineno}, column {exc.colno}")
    except OSError as exc:
        errors.append(f"Could not read diagnostic JSON {path}: {exc}")
    return None


def is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool))


def type_name(types: Iterable[type]) -> str:
    names = []
    for typ in types:
        if typ is type(None):
            names.append("null")
        else:
            names.append(typ.__name__)
    return " or ".join(names)


def matches_expected_type(value: Any, expected_types: tuple[type, ...]) -> bool:
    if int in expected_types and is_int(value):
        return True
    remaining = tuple(typ for typ in expected_types if typ is not int)
    return bool(remaining) and isinstance(value, remaining)


def validate_required_fields(report: dict[str, Any], errors: list[str]) -> None:
    for field, expected_types in REQUIRED_FIELDS.items():
        if field not in report:
            errors.append(f"Missing required field: {field}")
            continue
        value = report[field]
        if not matches_expected_type(value, expected_types):
            errors.append(
                f"Field {field!r} must be {type_name(expected_types)}, got {type(value).__name__}"
            )


def validate_generated_at(value: Any, errors: list[str]) -> None:
    if not isinstance(value, str):
        return
    try:
        dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        errors.append("Field 'generated_at' must be an ISO-8601 timestamp")


def validate_modules(report: dict[str, Any], errors: list[str]) -> tuple[int | None, int | None, int | None]:
    total = report.get("total_modules")
    passed = report.get("passed")
    failed = report.get("failed")
    modules = report.get("modules")

    if not is_int(total) or not is_int(passed) or not is_int(failed) or not isinstance(modules, list):
        return None, None, None

    if total < 0 or passed < 0 or failed < 0:
        errors.append("Fields 'total_modules', 'passed', and 'failed' must be non-negative integers")
    if total != len(modules):
        errors.append(f"Field 'total_modules' is {total}, but modules contains {len(modules)} entries")
    if passed + failed != total:
        errors.append(f"Fields 'passed' + 'failed' must equal 'total_modules' ({passed} + {failed} != {total})")

    counted_passed = 0
    counted_failed = 0
    for index, module in enumerate(modules):
        if not isinstance(module, dict):
            errors.append(f"modules[{index}] must be an object")
            continue

        for field, expected_types in MODULE_FIELDS.items():
            if field not in module:
                errors.append(f"modules[{index}] missing required field: {field}")
                continue
            value = module[field]
            if field == "elapsed_seconds":
                if not is_number(value):
                    errors.append(f"modules[{index}].elapsed_seconds must be a number")
                elif value < 0:
                    errors.append(f"modules[{index}].elapsed_seconds must be non-negative")
            elif not isinstance(value, expected_types):
                if matches_expected_type(value, expected_types):
                    continue
                errors.append(
                    f"modules[{index}].{field} must be {type_name(expected_types)}, got {type(value).__name__}"
                )

        status = module.get("status")
        if status == "PASS":
            counted_passed += 1
        elif status == "FAIL":
            counted_failed += 1
        elif isinstance(status, str):
            errors.append(f"modules[{index}].status must be PASS or FAIL, got {status!r}")

        name = module.get("name")
        if isinstance(name, str) and not name.strip():
            errors.append(f"modules[{index}].name must not be empty")

    if counted_passed != passed:
        errors.append(f"Field 'passed' is {passed}, but {counted_passed} module(s) have status PASS")
    if counted_failed != failed:
        errors.append(f"Field 'failed' is {failed}, but {counted_failed} module(s) have status FAIL")

    return total, passed, failed


def normalize_logd_paths(value: Any, errors: list[str]) -> list[str]:
    if isinstance(value, str):
        if not value.strip():
            errors.append("Field 'diagnostic_logd' must not be an empty string")
            return []
        return [value]
    if isinstance(value, list):
        if not value:
            errors.append("Field 'diagnostic_logd' must not be an empty list")
            return []
        paths: list[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                errors.append(f"diagnostic_logd[{index}] must be a non-empty string")
            else:
                paths.append(item)
        return paths
    if value is None:
        return []
    errors.append("Field 'diagnostic_logd' must be a string path, list of paths, or null")
    return []


def validate_logd_paths(
    report: dict[str, Any],
    report_path: Path,
    repo_root: Path,
    errors: list[str],
    warnings: list[str],
) -> None:
    diagnostic_error = report.get("diagnostic_logd_error")
    logd_paths = normalize_logd_paths(report.get("diagnostic_logd"), errors)
    commit = report.get("commit")

    if diagnostic_error:
        errors.append(f"Build script reported diagnostic_logd_error: {diagnostic_error}")
        return

    if not logd_paths:
        errors.append("Field 'diagnostic_logd' must reference at least one .logd artifact")
        return

    password = report.get("password")
    decrypt_command = report.get("decrypt_command")
    if not isinstance(password, str) or not password:
        errors.append("Field 'password' is required when diagnostic_logd is present")
    if not isinstance(decrypt_command, str) or not decrypt_command:
        errors.append("Field 'decrypt_command' is required when diagnostic_logd is present")

    if report.get("chunked") is True and len(logd_paths) == 1:
        warnings.append("Field 'chunked' is true but diagnostic_logd contains only one path")
    if report.get("chunked") is False and len(logd_paths) > 1:
        errors.append("Field 'chunked' must be true when diagnostic_logd contains multiple paths")

    expected_parts = []
    if len(logd_paths) > 1 and isinstance(commit, str):
        expected_parts = [f"diagnostic/build-{commit}-part{i:03d}.logd" for i in range(1, len(logd_paths) + 1)]
        if logd_paths != expected_parts:
            errors.append(
                "Chunked diagnostic_logd paths must be contiguous and ordered: "
                + ", ".join(expected_parts)
            )

    for ref in logd_paths:
        path_obj = Path(ref)
        if path_obj.is_absolute():
            errors.append(f"diagnostic_logd path must be repo-relative, got absolute path: {ref}")
            continue

        match = LOGD_RE.fullmatch(ref)
        if not match:
            errors.append(f"diagnostic_logd path has unexpected format: {ref}")
        elif isinstance(commit, str) and match.group("commit") != commit:
            errors.append(f"diagnostic_logd path {ref} does not match commit {commit}")

        resolved = (repo_root / ref).resolve()
        if not resolved.exists():
            fallback = (report_path.parent / path_obj.name).resolve()
            if fallback.exists():
                warnings.append(f"diagnostic_logd {ref} resolved beside the JSON file instead of repo root")
                resolved = fallback
            else:
                errors.append(f"Referenced diagnostic_logd file does not exist: {ref}")
                continue
        try:
            if resolved.stat().st_size <= 0:
                errors.append(f"Referenced diagnostic_logd file is empty: {ref}")
        except OSError as exc:
            errors.append(f"Could not stat diagnostic_logd file {ref}: {exc}")


def validate_report(path: Path, threshold: int = 0, verbose: bool = False) -> VerificationResult:
    errors: list[str] = []
    warnings: list[str] = []
    report_path = path.resolve()

    repo_root, git_warning = find_repo_root(report_path.parent)
    if verbose and git_warning:
        warnings.append(git_warning)

    report = read_json_report(report_path, errors)
    if report is None:
        return VerificationResult(str(report_path), False, threshold, errors=errors, warnings=warnings)
    if not isinstance(report, dict):
        errors.append("Diagnostic JSON root must be an object")
        return VerificationResult(str(report_path), False, threshold, errors=errors, warnings=warnings)

    validate_required_fields(report, errors)
    validate_generated_at(report.get("generated_at"), errors)

    commit = report.get("commit")
    if isinstance(commit, str) and not COMMIT_RE.fullmatch(commit):
        errors.append("Field 'commit' must be exactly 8 lowercase hex characters")

    if "chunk_size_bytes" in report:
        chunk_size = report.get("chunk_size_bytes")
        if chunk_size is not None and (not is_int(chunk_size) or chunk_size <= 0):
            errors.append("Field 'chunk_size_bytes' must be a positive integer or null")

    total, passed, failed = validate_modules(report, errors)
    validate_logd_paths(report, report_path, repo_root, errors, warnings)

    if passed is not None and passed < threshold:
        errors.append(f"Passing modules {passed} is below threshold {threshold}")

    modules = report.get("modules") if isinstance(report.get("modules"), list) else None
    return VerificationResult(
        path=str(report_path),
        ok=not errors,
        threshold=threshold,
        total_modules=total,
        passed=passed,
        failed=failed,
        errors=errors,
        warnings=warnings,
        modules=modules,
    )


def non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate build diagnostic JSON and referenced .logd artifacts."
    )
    parser.add_argument(
        "report",
        nargs="?",
        help="Path to diagnostic/build-<commit>.json. Defaults to the latest report under ./diagnostic.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print module details and non-fatal warnings")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    parser.add_argument(
        "--threshold",
        type=non_negative_int,
        default=0,
        help="Minimum number of passing modules required (default: 0)",
    )
    return parser.parse_args(argv)


def print_human(result: VerificationResult, verbose: bool = False) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"Diagnostic verification: {status}")
    print(f"Report: {result.path}")
    if result.total_modules is not None:
        print(
            f"Modules: {result.passed}/{result.total_modules} passed, "
            f"{result.failed} failed; threshold={result.threshold}"
        )

    for warning in result.warnings or []:
        print(f"WARNING: {warning}", file=sys.stderr)
    for error in result.errors or []:
        print(f"ERROR: {error}", file=sys.stderr)

    if verbose and result.modules:
        print("Module results:")
        for module in result.modules:
            name = module.get("name", "<unknown>")
            status = module.get("status", "<unknown>")
            elapsed = module.get("elapsed_seconds", "?")
            print(f"  - {name}: {status} ({elapsed}s)")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.report:
        report_path = Path(args.report)
    else:
        repo_root, _ = find_repo_root(Path.cwd())
        discovered = latest_diagnostic_report(repo_root)
        if discovered is None:
            result = VerificationResult(
                path=str(repo_root / "diagnostic"),
                ok=False,
                threshold=args.threshold,
                errors=["No diagnostic/build-*.json file found. Run `python3 build.py` first."],
                warnings=[],
            )
            if args.json:
                print(json.dumps(result.to_dict(verbose=args.verbose), indent=2))
            else:
                print_human(result, verbose=args.verbose)
            return 1
        report_path = discovered

    result = validate_report(report_path, threshold=args.threshold, verbose=args.verbose)
    if args.json:
        print(json.dumps(result.to_dict(verbose=args.verbose), indent=2))
    else:
        print_human(result, verbose=args.verbose)
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Validate build diagnostic metadata emitted by build.py.

The PR workflow validates diagnostic bundles after they are pushed. This tool
lets contributors catch malformed diagnostic JSON, missing .logd references,
and module pass-count threshold failures before opening a PR.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


COMMIT_RE = re.compile(r"^[0-9a-f]{8}$")
LOGD_RE = re.compile(r"^diagnostic/build-(?P<commit>[0-9a-f]{8})(?:-part(?P<part>\d{3}))?\.logd$")
REQUIRED_FIELDS: dict[str, tuple[type, ...]] = {
    "generated_at": (str,),
    "commit": (str,),
    "diagnostic_logd": (str, list, type(None)),
    "diagnostic_logd_error": (str, type(None)),
    "chunked": (bool,),
    "chunk_size_bytes": (int, type(None)),
    "password": (str, type(None)),
    "decrypt_command": (str, type(None)),
    "total_modules": (int,),
    "passed": (int,),
    "failed": (int,),
    "modules": (list,),
    "pr_note": (str,),
}

MODULE_FIELDS: dict[str, tuple[type, ...]] = {
    "name": (str,),
    "status": (str,),
    "elapsed_seconds": (int, float),
    "artifact": (str, type(None)),
    "output": (str,),
}


@dataclass
class CommandResult:
    ok: bool
    command: list[str]
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


@dataclass
class VerificationResult:
    path: str
    ok: bool
    threshold: int
    total_modules: int | None = None
    passed: int | None = None
    failed: int | None = None
    errors: list[str] | None = None
    warnings: list[str] | None = None
    modules: list[dict[str, Any]] | None = None

    def to_dict(self, verbose: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "path": self.path,
            "threshold": self.threshold,
            "total_modules": self.total_modules,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors or [],
            "warnings": self.warnings or [],
        }
        if verbose:
            payload["modules"] = self.modules or []
        return payload


def run_command(args: Sequence[str], cwd: Path | None = None, timeout: int = 10) -> CommandResult:
    """Run an external command and return a structured error instead of raising."""
    command = list(args)
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        return CommandResult(
            ok=False,
            command=command,
            returncode=None,
            error=f"Could not run {' '.join(command)!r}: executable not found ({exc}).",
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return CommandResult(
            ok=False,
            command=command,
            returncode=None,
            stdout=stdout,
            stderr=stderr,
            error=f"Command {' '.join(command)!r} timed out after {timeout}s.",
        )
    except OSError as exc:
        return CommandResult(
            ok=False,
            command=command,
            returncode=None,
            error=f"Could not run {' '.join(command)!r}: {exc}.",
        )

    stderr = completed.stderr.strip()
    stdout = completed.stdout.strip()
    if completed.returncode != 0:
        detail = stderr or stdout or "no output"
        return CommandResult(
            ok=False,
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            error=f"Command {' '.join(command)!r} exited {completed.returncode}: {detail}",
        )

    return CommandResult(
        ok=True,
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def find_repo_root(start: Path) -> tuple[Path, str | None]:
    """Find a repo root with git when available, then fall back to parent search."""
    start_dir = start if start.is_dir() else start.parent
    git_result = run_command(["git", "rev-parse", "--show-toplevel"], cwd=start_dir, timeout=5)
    if git_result.ok and git_result.stdout.strip():
        return Path(git_result.stdout.strip()).resolve(), None

    for candidate in (start_dir.resolve(), *start_dir.resolve().parents):
        if (candidate / ".git").exists():
            return candidate, git_result.error

    if start_dir.name == "diagnostic":
        return start_dir.parent.resolve(), git_result.error
    return start_dir.resolve(), git_result.error


def latest_diagnostic_report(repo_root: Path) -> Path | None:
    diagnostic_dir = repo_root / "diagnostic"
    if not diagnostic_dir.exists():
        return None
    reports = sorted(
        diagnostic_dir.glob("build-*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return reports[0] if reports else None


def read_json_report(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(f"Diagnostic JSON not found: {path}")
    except json.JSONDecodeError as exc:
        errors.append(f"Diagnostic JSON is not valid JSON: {exc.msg} at line {exc.lineno}, column {exc.colno}")
    except OSError as exc:
        errors.append(f"Could not read diagnostic JSON {path}: {exc}")
    return None


def is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool))


def type_name(types: Iterable[type]) -> str:
    names = []
    for typ in types:
        if typ is type(None):
            names.append("null")
        else:
            names.append(typ.__name__)
    return " or ".join(names)


def matches_expected_type(value: Any, expected_types: tuple[type, ...]) -> bool:
    if int in expected_types and is_int(value):
        return True
    remaining = tuple(typ for typ in expected_types if typ is not int)
    return bool(remaining) and isinstance(value, remaining)


def validate_required_fields(report: dict[str, Any], errors: list[str]) -> None:
    for field, expected_types in REQUIRED_FIELDS.items():
        if field not in report:
            errors.append(f"Missing required field: {field}")
            continue
        value = report[field]
        if not matches_expected_type(value, expected_types):
            errors.append(
                f"Field {field!r} must be {type_name(expected_types)}, got {type(value).__name__}"
            )


def validate_generated_at(value: Any, errors: list[str]) -> None:
    if not isinstance(value, str):
        return
    try:
        dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        errors.append("Field 'generated_at' must be an ISO-8601 timestamp")


def validate_modules(report: dict[str, Any], errors: list[str]) -> tuple[int | None, int | None, int | None]:
    total = report.get("total_modules")
    passed = report.get("passed")
    failed = report.get("failed")
    modules = report.get("modules")

    if not is_int(total) or not is_int(passed) or not is_int(failed) or not isinstance(modules, list):
        return None, None, None

    if total < 0 or passed < 0 or failed < 0:
        errors.append("Fields 'total_modules', 'passed', and 'failed' must be non-negative integers")
    if total != len(modules):
        errors.append(f"Field 'total_modules' is {total}, but modules contains {len(modules)} entries")
    if passed + failed != total:
        errors.append(f"Fields 'passed' + 'failed' must equal 'total_modules' ({passed} + {failed} != {total})")

    counted_passed = 0
    counted_failed = 0
    for index, module in enumerate(modules):
        if not isinstance(module, dict):
            errors.append(f"modules[{index}] must be an object")
            continue

        for field, expected_types in MODULE_FIELDS.items():
            if field not in module:
                errors.append(f"modules[{index}] missing required field: {field}")
                continue
            value = module[field]
            if field == "elapsed_seconds":
                if not is_number(value):
                    errors.append(f"modules[{index}].elapsed_seconds must be a number")
                elif value < 0:
                    errors.append(f"modules[{index}].elapsed_seconds must be non-negative")
            elif not isinstance(value, expected_types):
                if matches_expected_type(value, expected_types):
                    continue
                errors.append(
                    f"modules[{index}].{field} must be {type_name(expected_types)}, got {type(value).__name__}"
                )

        status = module.get("status")
        if status == "PASS":
            counted_passed += 1
        elif status == "FAIL":
            counted_failed += 1
        elif isinstance(status, str):
            errors.append(f"modules[{index}].status must be PASS or FAIL, got {status!r}")

        name = module.get("name")
        if isinstance(name, str) and not name.strip():
            errors.append(f"modules[{index}].name must not be empty")

    if counted_passed != passed:
        errors.append(f"Field 'passed' is {passed}, but {counted_passed} module(s) have status PASS")
    if counted_failed != failed:
        errors.append(f"Field 'failed' is {failed}, but {counted_failed} module(s) have status FAIL")

    return total, passed, failed


def normalize_logd_paths(value: Any, errors: list[str]) -> list[str]:
    if isinstance(value, str):
        if not value.strip():
            errors.append("Field 'diagnostic_logd' must not be an empty string")
            return []
        return [value]
    if isinstance(value, list):
        if not value:
            errors.append("Field 'diagnostic_logd' must not be an empty list")
            return []
        paths: list[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                errors.append(f"diagnostic_logd[{index}] must be a non-empty string")
            else:
                paths.append(item)
        return paths
    if value is None:
        return []
    errors.append("Field 'diagnostic_logd' must be a string path, list of paths, or null")
    return []


def validate_logd_paths(
    report: dict[str, Any],
    report_path: Path,
    repo_root: Path,
    errors: list[str],
    warnings: list[str],
) -> None:
    diagnostic_error = report.get("diagnostic_logd_error")
    logd_paths = normalize_logd_paths(report.get("diagnostic_logd"), errors)
    commit = report.get("commit")

    if diagnostic_error:
        errors.append(f"Build script reported diagnostic_logd_error: {diagnostic_error}")
        return

    if not logd_paths:
        errors.append("Field 'diagnostic_logd' must reference at least one .logd artifact")
        return

    password = report.get("password")
    decrypt_command = report.get("decrypt_command")
    if not isinstance(password, str) or not password:
        errors.append("Field 'password' is required when diagnostic_logd is present")
    if not isinstance(decrypt_command, str) or not decrypt_command:
        errors.append("Field 'decrypt_command' is required when diagnostic_logd is present")

    if report.get("chunked") is True and len(logd_paths) == 1:
        warnings.append("Field 'chunked' is true but diagnostic_logd contains only one path")
    if report.get("chunked") is False and len(logd_paths) > 1:
        errors.append("Field 'chunked' must be true when diagnostic_logd contains multiple paths")

    expected_parts = []
    if len(logd_paths) > 1 and isinstance(commit, str):
        expected_parts = [f"diagnostic/build-{commit}-part{i:03d}.logd" for i in range(1, len(logd_paths) + 1)]
        if logd_paths != expected_parts:
            errors.append(
                "Chunked diagnostic_logd paths must be contiguous and ordered: "
                + ", ".join(expected_parts)
            )

    for ref in logd_paths:
        path_obj = Path(ref)
        if path_obj.is_absolute():
            errors.append(f"diagnostic_logd path must be repo-relative, got absolute path: {ref}")
            continue

        match = LOGD_RE.fullmatch(ref)
        if not match:
            errors.append(f"diagnostic_logd path has unexpected format: {ref}")
        elif isinstance(commit, str) and match.group("commit") != commit:
            errors.append(f"diagnostic_logd path {ref} does not match commit {commit}")

        resolved = (repo_root / ref).resolve()
        if not resolved.exists():
            fallback = (report_path.parent / path_obj.name).resolve()
            if fallback.exists():
                warnings.append(f"diagnostic_logd {ref} resolved beside the JSON file instead of repo root")
                resolved = fallback
            else:
                errors.append(f"Referenced diagnostic_logd file does not exist: {ref}")
                continue
        try:
            if resolved.stat().st_size <= 0:
                errors.append(f"Referenced diagnostic_logd file is empty: {ref}")
        except OSError as exc:
            errors.append(f"Could not stat diagnostic_logd file {ref}: {exc}")


def validate_report(path: Path, threshold: int = 0, verbose: bool = False) -> VerificationResult:
    errors: list[str] = []
    warnings: list[str] = []
    report_path = path.resolve()

    repo_root, git_warning = find_repo_root(report_path.parent)
    if verbose and git_warning:
        warnings.append(git_warning)

    report = read_json_report(report_path, errors)
    if report is None:
        return VerificationResult(str(report_path), False, threshold, errors=errors, warnings=warnings)
    if not isinstance(report, dict):
        errors.append("Diagnostic JSON root must be an object")
        return VerificationResult(str(report_path), False, threshold, errors=errors, warnings=warnings)

    validate_required_fields(report, errors)
    validate_generated_at(report.get("generated_at"), errors)

    commit = report.get("commit")
    if isinstance(commit, str) and not COMMIT_RE.fullmatch(commit):
        errors.append("Field 'commit' must be exactly 8 lowercase hex characters")

    if "chunk_size_bytes" in report:
        chunk_size = report.get("chunk_size_bytes")
        if chunk_size is not None and (not is_int(chunk_size) or chunk_size <= 0):
            errors.append("Field 'chunk_size_bytes' must be a positive integer or null")

    total, passed, failed = validate_modules(report, errors)
    validate_logd_paths(report, report_path, repo_root, errors, warnings)

    if passed is not None and passed < threshold:
        errors.append(f"Passing modules {passed} is below threshold {threshold}")

    modules = report.get("modules") if isinstance(report.get("modules"), list) else None
    return VerificationResult(
        path=str(report_path),
        ok=not errors,
        threshold=threshold,
        total_modules=total,
        passed=passed,
        failed=failed,
        errors=errors,
        warnings=warnings,
        modules=modules,
    )


def non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate build diagnostic JSON and referenced .logd artifacts."
    )
    parser.add_argument(
        "report",
        nargs="?",
        help="Path to diagnostic/build-<commit>.json. Defaults to the latest report under ./diagnostic.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print module details and non-fatal warnings")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    parser.add_argument(
        "--threshold",
        type=non_negative_int,
        default=0,
        help="Minimum number of passing modules required (default: 0)",
    )
    return parser.parse_args(argv)


def print_human(result: VerificationResult, verbose: bool = False) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"Diagnostic verification: {status}")
    print(f"Report: {result.path}")
    if result.total_modules is not None:
        print(
            f"Modules: {result.passed}/{result.total_modules} passed, "
            f"{result.failed} failed; threshold={result.threshold}"
        )

    for warning in result.warnings or []:
        print(f"WARNING: {warning}", file=sys.stderr)
    for error in result.errors or []:
        print(f"ERROR: {error}", file=sys.stderr)

    if verbose and result.modules:
        print("Module results:")
        for module in result.modules:
            name = module.get("name", "<unknown>")
            status = module.get("status", "<unknown>")
            elapsed = module.get("elapsed_seconds", "?")
            print(f"  - {name}: {status} ({elapsed}s)")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.report:
        report_path = Path(args.report)
    else:
        repo_root, _ = find_repo_root(Path.cwd())
        discovered = latest_diagnostic_report(repo_root)
        if discovered is None:
            result = VerificationResult(
                path=str(repo_root / "diagnostic"),
                ok=False,
                threshold=args.threshold,
                errors=["No diagnostic/build-*.json file found. Run `python3 build.py` first."],
                warnings=[],
            )
            if args.json:
                print(json.dumps(result.to_dict(verbose=args.verbose), indent=2))
            else:
                print_human(result, verbose=args.verbose)
            return 1
        report_path = discovered

    result = validate_report(report_path, threshold=args.threshold, verbose=args.verbose)
    if args.json:
        print(json.dumps(result.to_dict(verbose=args.verbose), indent=2))
    else:
        print_human(result, verbose=args.verbose)
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Validate build diagnostic metadata emitted by build.py.

The PR workflow validates diagnostic bundles after they are pushed. This tool
lets contributors catch malformed diagnostic JSON, missing .logd references,
and module pass-count threshold failures before opening a PR.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


COMMIT_RE = re.compile(r"^[0-9a-f]{8}$")
LOGD_RE = re.compile(r"^diagnostic/build-(?P<commit>[0-9a-f]{8})(?:-part(?P<part>\d{3}))?\.logd$")
REQUIRED_FIELDS: dict[str, tuple[type, ...]] = {
    "generated_at": (str,),
    "commit": (str,),
    "diagnostic_logd": (str, list, type(None)),
    "diagnostic_logd_error": (str, type(None)),
    "chunked": (bool,),
    "chunk_size_bytes": (int, type(None)),
    "password": (str, type(None)),
    "decrypt_command": (str, type(None)),
    "total_modules": (int,),
    "passed": (int,),
    "failed": (int,),
    "modules": (list,),
    "pr_note": (str,),
}

MODULE_FIELDS: dict[str, tuple[type, ...]] = {
    "name": (str,),
    "status": (str,),
    "elapsed_seconds": (int, float),
    "artifact": (str, type(None)),
    "output": (str,),
}


@dataclass
class CommandResult:
    ok: bool
    command: list[str]
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


@dataclass
class VerificationResult:
    path: str
    ok: bool
    threshold: int
    total_modules: int | None = None
    passed: int | None = None
    failed: int | None = None
    errors: list[str] | None = None
    warnings: list[str] | None = None
    modules: list[dict[str, Any]] | None = None

    def to_dict(self, verbose: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "path": self.path,
            "threshold": self.threshold,
            "total_modules": self.total_modules,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors or [],
            "warnings": self.warnings or [],
        }
        if verbose:
            payload["modules"] = self.modules or []
        return payload


def run_command(args: Sequence[str], cwd: Path | None = None, timeout: int = 10) -> CommandResult:
    """Run an external command and return a structured error instead of raising."""
    command = list(args)
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        return CommandResult(
            ok=False,
            command=command,
            returncode=None,
            error=f"Could not run {' '.join(command)!r}: executable not found ({exc}).",
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return CommandResult(
            ok=False,
            command=command,
            returncode=None,
            stdout=stdout,
            stderr=stderr,
            error=f"Command {' '.join(command)!r} timed out after {timeout}s.",
        )
    except OSError as exc:
        return CommandResult(
            ok=False,
            command=command,
            returncode=None,
            error=f"Could not run {' '.join(command)!r}: {exc}.",
        )

    stderr = completed.stderr.strip()
    stdout = completed.stdout.strip()
    if completed.returncode != 0:
        detail = stderr or stdout or "no output"
        return CommandResult(
            ok=False,
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            error=f"Command {' '.join(command)!r} exited {completed.returncode}: {detail}",
        )

    return CommandResult(
        ok=True,
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def find_repo_root(start: Path) -> tuple[Path, str | None]:
    """Find a repo root with git when available, then fall back to parent search."""
    start_dir = start if start.is_dir() else start.parent
    git_result = run_command(["git", "rev-parse", "--show-toplevel"], cwd=start_dir, timeout=5)
    if git_result.ok and git_result.stdout.strip():
        return Path(git_result.stdout.strip()).resolve(), None

    for candidate in (start_dir.resolve(), *start_dir.resolve().parents):
        if (candidate / ".git").exists():
            return candidate, git_result.error

    if start_dir.name == "diagnostic":
        return start_dir.parent.resolve(), git_result.error
    return start_dir.resolve(), git_result.error


def latest_diagnostic_report(repo_root: Path) -> Path | None:
    diagnostic_dir = repo_root / "diagnostic"
    if not diagnostic_dir.exists():
        return None
    reports = sorted(
        diagnostic_dir.glob("build-*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return reports[0] if reports else None


def read_json_report(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(f"Diagnostic JSON not found: {path}")
    except json.JSONDecodeError as exc:
        errors.append(f"Diagnostic JSON is not valid JSON: {exc.msg} at line {exc.lineno}, column {exc.colno}")
    except OSError as exc:
        errors.append(f"Could not read diagnostic JSON {path}: {exc}")
    return None


def is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool))


def type_name(types: Iterable[type]) -> str:
    names = []
    for typ in types:
        if typ is type(None):
            names.append("null")
        else:
            names.append(typ.__name__)
    return " or ".join(names)


def matches_expected_type(value: Any, expected_types: tuple[type, ...]) -> bool:
    if int in expected_types and is_int(value):
        return True
    remaining = tuple(typ for typ in expected_types if typ is not int)
    return bool(remaining) and isinstance(value, remaining)


def validate_required_fields(report: dict[str, Any], errors: list[str]) -> None:
    for field, expected_types in REQUIRED_FIELDS.items():
        if field not in report:
            errors.append(f"Missing required field: {field}")
            continue
        value = report[field]
        if not matches_expected_type(value, expected_types):
            errors.append(
                f"Field {field!r} must be {type_name(expected_types)}, got {type(value).__name__}"
            )


def validate_generated_at(value: Any, errors: list[str]) -> None:
    if not isinstance(value, str):
        return
    try:
        dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        errors.append("Field 'generated_at' must be an ISO-8601 timestamp")


def validate_modules(report: dict[str, Any], errors: list[str]) -> tuple[int | None, int | None, int | None]:
    total = report.get("total_modules")
    passed = report.get("passed")
    failed = report.get("failed")
    modules = report.get("modules")

    if not is_int(total) or not is_int(passed) or not is_int(failed) or not isinstance(modules, list):
        return None, None, None

    if total < 0 or passed < 0 or failed < 0:
        errors.append("Fields 'total_modules', 'passed', and 'failed' must be non-negative integers")
    if total != len(modules):
        errors.append(f"Field 'total_modules' is {total}, but modules contains {len(modules)} entries")
    if passed + failed != total:
        errors.append(f"Fields 'passed' + 'failed' must equal 'total_modules' ({passed} + {failed} != {total})")

    counted_passed = 0
    counted_failed = 0
    for index, module in enumerate(modules):
        if not isinstance(module, dict):
            errors.append(f"modules[{index}] must be an object")
            continue

        for field, expected_types in MODULE_FIELDS.items():
            if field not in module:
                errors.append(f"modules[{index}] missing required field: {field}")
                continue
            value = module[field]
            if field == "elapsed_seconds":
                if not is_number(value):
                    errors.append(f"modules[{index}].elapsed_seconds must be a number")
                elif value < 0:
                    errors.append(f"modules[{index}].elapsed_seconds must be non-negative")
            elif not isinstance(value, expected_types):
                if matches_expected_type(value, expected_types):
                    continue
                errors.append(
                    f"modules[{index}].{field} must be {type_name(expected_types)}, got {type(value).__name__}"
                )

        status = module.get("status")
        if status == "PASS":
            counted_passed += 1
        elif status == "FAIL":
            counted_failed += 1
        elif isinstance(status, str):
            errors.append(f"modules[{index}].status must be PASS or FAIL, got {status!r}")

        name = module.get("name")
        if isinstance(name, str) and not name.strip():
            errors.append(f"modules[{index}].name must not be empty")

    if counted_passed != passed:
        errors.append(f"Field 'passed' is {passed}, but {counted_passed} module(s) have status PASS")
    if counted_failed != failed:
        errors.append(f"Field 'failed' is {failed}, but {counted_failed} module(s) have status FAIL")

    return total, passed, failed


def normalize_logd_paths(value: Any, errors: list[str]) -> list[str]:
    if isinstance(value, str):
        if not value.strip():
            errors.append("Field 'diagnostic_logd' must not be an empty string")
            return []
        return [value]
    if isinstance(value, list):
        if not value:
            errors.append("Field 'diagnostic_logd' must not be an empty list")
            return []
        paths: list[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                errors.append(f"diagnostic_logd[{index}] must be a non-empty string")
            else:
                paths.append(item)
        return paths
    if value is None:
        return []
    errors.append("Field 'diagnostic_logd' must be a string path, list of paths, or null")
    return []


def validate_logd_paths(
    report: dict[str, Any],
    report_path: Path,
    repo_root: Path,
    errors: list[str],
    warnings: list[str],
) -> None:
    diagnostic_error = report.get("diagnostic_logd_error")
    logd_paths = normalize_logd_paths(report.get("diagnostic_logd"), errors)
    commit = report.get("commit")

    if diagnostic_error:
        errors.append(f"Build script reported diagnostic_logd_error: {diagnostic_error}")
        return

    if not logd_paths:
        errors.append("Field 'diagnostic_logd' must reference at least one .logd artifact")
        return

    password = report.get("password")
    decrypt_command = report.get("decrypt_command")
    if not isinstance(password, str) or not password:
        errors.append("Field 'password' is required when diagnostic_logd is present")
    if not isinstance(decrypt_command, str) or not decrypt_command:
        errors.append("Field 'decrypt_command' is required when diagnostic_logd is present")

    if report.get("chunked") is True and len(logd_paths) == 1:
        warnings.append("Field 'chunked' is true but diagnostic_logd contains only one path")
    if report.get("chunked") is False and len(logd_paths) > 1:
        errors.append("Field 'chunked' must be true when diagnostic_logd contains multiple paths")

    expected_parts = []
    if len(logd_paths) > 1 and isinstance(commit, str):
        expected_parts = [f"diagnostic/build-{commit}-part{i:03d}.logd" for i in range(1, len(logd_paths) + 1)]
        if logd_paths != expected_parts:
            errors.append(
                "Chunked diagnostic_logd paths must be contiguous and ordered: "
                + ", ".join(expected_parts)
            )

    for ref in logd_paths:
        path_obj = Path(ref)
        if path_obj.is_absolute():
            errors.append(f"diagnostic_logd path must be repo-relative, got absolute path: {ref}")
            continue

        match = LOGD_RE.fullmatch(ref)
        if not match:
            errors.append(f"diagnostic_logd path has unexpected format: {ref}")
        elif isinstance(commit, str) and match.group("commit") != commit:
            errors.append(f"diagnostic_logd path {ref} does not match commit {commit}")

        resolved = (repo_root / ref).resolve()
        if not resolved.exists():
            fallback = (report_path.parent / path_obj.name).resolve()
            if fallback.exists():
                warnings.append(f"diagnostic_logd {ref} resolved beside the JSON file instead of repo root")
                resolved = fallback
            else:
                errors.append(f"Referenced diagnostic_logd file does not exist: {ref}")
                continue
        try:
            if resolved.stat().st_size <= 0:
                errors.append(f"Referenced diagnostic_logd file is empty: {ref}")
        except OSError as exc:
            errors.append(f"Could not stat diagnostic_logd file {ref}: {exc}")


def validate_report(path: Path, threshold: int = 0, verbose: bool = False) -> VerificationResult:
    errors: list[str] = []
    warnings: list[str] = []
    report_path = path.resolve()

    repo_root, git_warning = find_repo_root(report_path.parent)
    if verbose and git_warning:
        warnings.append(git_warning)

    report = read_json_report(report_path, errors)
    if report is None:
        return VerificationResult(str(report_path), False, threshold, errors=errors, warnings=warnings)
    if not isinstance(report, dict):
        errors.append("Diagnostic JSON root must be an object")
        return VerificationResult(str(report_path), False, threshold, errors=errors, warnings=warnings)

    validate_required_fields(report, errors)
    validate_generated_at(report.get("generated_at"), errors)

    commit = report.get("commit")
    if isinstance(commit, str) and not COMMIT_RE.fullmatch(commit):
        errors.append("Field 'commit' must be exactly 8 lowercase hex characters")

    if "chunk_size_bytes" in report:
        chunk_size = report.get("chunk_size_bytes")
        if chunk_size is not None and (not is_int(chunk_size) or chunk_size <= 0):
            errors.append("Field 'chunk_size_bytes' must be a positive integer or null")

    total, passed, failed = validate_modules(report, errors)
    validate_logd_paths(report, report_path, repo_root, errors, warnings)

    if passed is not None and passed < threshold:
        errors.append(f"Passing modules {passed} is below threshold {threshold}")

    modules = report.get("modules") if isinstance(report.get("modules"), list) else None
    return VerificationResult(
        path=str(report_path),
        ok=not errors,
        threshold=threshold,
        total_modules=total,
        passed=passed,
        failed=failed,
        errors=errors,
        warnings=warnings,
        modules=modules,
    )


def non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate build diagnostic JSON and referenced .logd artifacts."
    )
    parser.add_argument(
        "report",
        nargs="?",
        help="Path to diagnostic/build-<commit>.json. Defaults to the latest report under ./diagnostic.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print module details and non-fatal warnings")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    parser.add_argument(
        "--threshold",
        type=non_negative_int,
        default=0,
        help="Minimum number of passing modules required (default: 0)",
    )
    return parser.parse_args(argv)


def print_human(result: VerificationResult, verbose: bool = False) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"Diagnostic verification: {status}")
    print(f"Report: {result.path}")
    if result.total_modules is not None:
        print(
            f"Modules: {result.passed}/{result.total_modules} passed, "
            f"{result.failed} failed; threshold={result.threshold}"
        )

    for warning in result.warnings or []:
        print(f"WARNING: {warning}", file=sys.stderr)
    for error in result.errors or []:
        print(f"ERROR: {error}", file=sys.stderr)

    if verbose and result.modules:
        print("Module results:")
        for module in result.modules:
            name = module.get("name", "<unknown>")
            status = module.get("status", "<unknown>")
            elapsed = module.get("elapsed_seconds", "?")
            print(f"  - {name}: {status} ({elapsed}s)")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.report:
        report_path = Path(args.report)
    else:
        repo_root, _ = find_repo_root(Path.cwd())
        discovered = latest_diagnostic_report(repo_root)
        if discovered is None:
            result = VerificationResult(
                path=str(repo_root / "diagnostic"),
                ok=False,
                threshold=args.threshold,
                errors=["No diagnostic/build-*.json file found. Run `python3 build.py` first."],
                warnings=[],
            )
            if args.json:
                print(json.dumps(result.to_dict(verbose=args.verbose), indent=2))
            else:
                print_human(result, verbose=args.verbose)
            return 1
        report_path = discovered

    result = validate_report(report_path, threshold=args.threshold, verbose=args.verbose)
    if args.json:
        print(json.dumps(result.to_dict(verbose=args.verbose), indent=2))
    else:
        print_human(result, verbose=args.verbose)
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Validate build diagnostic metadata emitted by build.py.

The PR workflow validates diagnostic bundles after they are pushed. This tool
lets contributors catch malformed diagnostic JSON, missing .logd references,
and module pass-count threshold failures before opening a PR.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


COMMIT_RE = re.compile(r"^[0-9a-f]{8}$")
LOGD_RE = re.compile(r"^diagnostic/build-(?P<commit>[0-9a-f]{8})(?:-part(?P<part>\d{3}))?\.logd$")
REQUIRED_FIELDS: dict[str, tuple[type, ...]] = {
    "generated_at": (str,),
    "commit": (str,),
    "diagnostic_logd": (str, list, type(None)),
    "diagnostic_logd_error": (str, type(None)),
    "chunked": (bool,),
    "chunk_size_bytes": (int, type(None)),
    "password": (str, type(None)),
    "decrypt_command": (str, type(None)),
    "total_modules": (int,),
    "passed": (int,),
    "failed": (int,),
    "modules": (list,),
    "pr_note": (str,),
}

MODULE_FIELDS: dict[str, tuple[type, ...]] = {
    "name": (str,),
    "status": (str,),
    "elapsed_seconds": (int, float),
    "artifact": (str, type(None)),
    "output": (str,),
}


@dataclass
class CommandResult:
    ok: bool
    command: list[str]
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


@dataclass
class VerificationResult:
    path: str
    ok: bool
    threshold: int
    total_modules: int | None = None
    passed: int | None = None
    failed: int | None = None
    errors: list[str] | None = None
    warnings: list[str] | None = None
    modules: list[dict[str, Any]] | None = None

    def to_dict(self, verbose: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "path": self.path,
            "threshold": self.threshold,
            "total_modules": self.total_modules,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors or [],
            "warnings": self.warnings or [],
        }
        if verbose:
            payload["modules"] = self.modules or []
        return payload


def run_command(args: Sequence[str], cwd: Path | None = None, timeout: int = 10) -> CommandResult:
    """Run an external command and return a structured error instead of raising."""
    command = list(args)
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        return CommandResult(
            ok=False,
            command=command,
            returncode=None,
            error=f"Could not run {' '.join(command)!r}: executable not found ({exc}).",
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return CommandResult(
            ok=False,
            command=command,
            returncode=None,
            stdout=stdout,
            stderr=stderr,
            error=f"Command {' '.join(command)!r} timed out after {timeout}s.",
        )
    except OSError as exc:
        return CommandResult(
            ok=False,
            command=command,
            returncode=None,
            error=f"Could not run {' '.join(command)!r}: {exc}.",
        )

    stderr = completed.stderr.strip()
    stdout = completed.stdout.strip()
    if completed.returncode != 0:
        detail = stderr or stdout or "no output"
        return CommandResult(
            ok=False,
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            error=f"Command {' '.join(command)!r} exited {completed.returncode}: {detail}",
        )

    return CommandResult(
        ok=True,
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def find_repo_root(start: Path) -> tuple[Path, str | None]:
    """Find a repo root with git when available, then fall back to parent search."""
    start_dir = start if start.is_dir() else start.parent
    git_result = run_command(["git", "rev-parse", "--show-toplevel"], cwd=start_dir, timeout=5)
    if git_result.ok and git_result.stdout.strip():
        return Path(git_result.stdout.strip()).resolve(), None

    for candidate in (start_dir.resolve(), *start_dir.resolve().parents):
        if (candidate / ".git").exists():
            return candidate, git_result.error

    if start_dir.name == "diagnostic":
        return start_dir.parent.resolve(), git_result.error
    return start_dir.resolve(), git_result.error


def latest_diagnostic_report(repo_root: Path) -> Path | None:
    diagnostic_dir = repo_root / "diagnostic"
    if not diagnostic_dir.exists():
        return None
    reports = sorted(
        diagnostic_dir.glob("build-*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return reports[0] if reports else None


def read_json_report(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(f"Diagnostic JSON not found: {path}")
    except json.JSONDecodeError as exc:
        errors.append(f"Diagnostic JSON is not valid JSON: {exc.msg} at line {exc.lineno}, column {exc.colno}")
    except OSError as exc:
        errors.append(f"Could not read diagnostic JSON {path}: {exc}")
    return None


def is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool))


def type_name(types: Iterable[type]) -> str:
    names = []
    for typ in types:
        if typ is type(None):
            names.append("null")
        else:
            names.append(typ.__name__)
    return " or ".join(names)


def matches_expected_type(value: Any, expected_types: tuple[type, ...]) -> bool:
    if int in expected_types and is_int(value):
        return True
    remaining = tuple(typ for typ in expected_types if typ is not int)
    return bool(remaining) and isinstance(value, remaining)


def validate_required_fields(report: dict[str, Any], errors: list[str]) -> None:
    for field, expected_types in REQUIRED_FIELDS.items():
        if field not in report:
            errors.append(f"Missing required field: {field}")
            continue
        value = report[field]
        if not matches_expected_type(value, expected_types):
            errors.append(
                f"Field {field!r} must be {type_name(expected_types)}, got {type(value).__name__}"
            )


def validate_generated_at(value: Any, errors: list[str]) -> None:
    if not isinstance(value, str):
        return
    try:
        dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        errors.append("Field 'generated_at' must be an ISO-8601 timestamp")


def validate_modules(report: dict[str, Any], errors: list[str]) -> tuple[int | None, int | None, int | None]:
    total = report.get("total_modules")
    passed = report.get("passed")
    failed = report.get("failed")
    modules = report.get("modules")

    if not is_int(total) or not is_int(passed) or not is_int(failed) or not isinstance(modules, list):
        return None, None, None

    if total < 0 or passed < 0 or failed < 0:
        errors.append("Fields 'total_modules', 'passed', and 'failed' must be non-negative integers")
    if total != len(modules):
        errors.append(f"Field 'total_modules' is {total}, but modules contains {len(modules)} entries")
    if passed + failed != total:
        errors.append(f"Fields 'passed' + 'failed' must equal 'total_modules' ({passed} + {failed} != {total})")

    counted_passed = 0
    counted_failed = 0
    for index, module in enumerate(modules):
        if not isinstance(module, dict):
            errors.append(f"modules[{index}] must be an object")
            continue

        for field, expected_types in MODULE_FIELDS.items():
            if field not in module:
                errors.append(f"modules[{index}] missing required field: {field}")
                continue
            value = module[field]
            if field == "elapsed_seconds":
                if not is_number(value):
                    errors.append(f"modules[{index}].elapsed_seconds must be a number")
                elif value < 0:
                    errors.append(f"modules[{index}].elapsed_seconds must be non-negative")
            elif not isinstance(value, expected_types):
                if matches_expected_type(value, expected_types):
                    continue
                errors.append(
                    f"modules[{index}].{field} must be {type_name(expected_types)}, got {type(value).__name__}"
                )

        status = module.get("status")
        if status == "PASS":
            counted_passed += 1
        elif status == "FAIL":
            counted_failed += 1
        elif isinstance(status, str):
            errors.append(f"modules[{index}].status must be PASS or FAIL, got {status!r}")

        name = module.get("name")
        if isinstance(name, str) and not name.strip():
            errors.append(f"modules[{index}].name must not be empty")

    if counted_passed != passed:
        errors.append(f"Field 'passed' is {passed}, but {counted_passed} module(s) have status PASS")
    if counted_failed != failed:
        errors.append(f"Field 'failed' is {failed}, but {counted_failed} module(s) have status FAIL")

    return total, passed, failed


def normalize_logd_paths(value: Any, errors: list[str]) -> list[str]:
    if isinstance(value, str):
        if not value.strip():
            errors.append("Field 'diagnostic_logd' must not be an empty string")
            return []
        return [value]
    if isinstance(value, list):
        if not value:
            errors.append("Field 'diagnostic_logd' must not be an empty list")
            return []
        paths: list[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                errors.append(f"diagnostic_logd[{index}] must be a non-empty string")
            else:
                paths.append(item)
        return paths
    if value is None:
        return []
    errors.append("Field 'diagnostic_logd' must be a string path, list of paths, or null")
    return []


def validate_logd_paths(
    report: dict[str, Any],
    report_path: Path,
    repo_root: Path,
    errors: list[str],
    warnings: list[str],
) -> None:
    diagnostic_error = report.get("diagnostic_logd_error")
    logd_paths = normalize_logd_paths(report.get("diagnostic_logd"), errors)
    commit = report.get("commit")

    if diagnostic_error:
        errors.append(f"Build script reported diagnostic_logd_error: {diagnostic_error}")
        return

    if not logd_paths:
        errors.append("Field 'diagnostic_logd' must reference at least one .logd artifact")
        return

    password = report.get("password")
    decrypt_command = report.get("decrypt_command")
    if not isinstance(password, str) or not password:
        errors.append("Field 'password' is required when diagnostic_logd is present")
    if not isinstance(decrypt_command, str) or not decrypt_command:
        errors.append("Field 'decrypt_command' is required when diagnostic_logd is present")

    if report.get("chunked") is True and len(logd_paths) == 1:
        warnings.append("Field 'chunked' is true but diagnostic_logd contains only one path")
    if report.get("chunked") is False and len(logd_paths) > 1:
        errors.append("Field 'chunked' must be true when diagnostic_logd contains multiple paths")

    expected_parts = []
    if len(logd_paths) > 1 and isinstance(commit, str):
        expected_parts = [f"diagnostic/build-{commit}-part{i:03d}.logd" for i in range(1, len(logd_paths) + 1)]
        if logd_paths != expected_parts:
            errors.append(
                "Chunked diagnostic_logd paths must be contiguous and ordered: "
                + ", ".join(expected_parts)
            )

    for ref in logd_paths:
        path_obj = Path(ref)
        if path_obj.is_absolute():
            errors.append(f"diagnostic_logd path must be repo-relative, got absolute path: {ref}")
            continue

        match = LOGD_RE.fullmatch(ref)
        if not match:
            errors.append(f"diagnostic_logd path has unexpected format: {ref}")
        elif isinstance(commit, str) and match.group("commit") != commit:
            errors.append(f"diagnostic_logd path {ref} does not match commit {commit}")

        resolved = (repo_root / ref).resolve()
        if not resolved.exists():
            fallback = (report_path.parent / path_obj.name).resolve()
            if fallback.exists():
                warnings.append(f"diagnostic_logd {ref} resolved beside the JSON file instead of repo root")
                resolved = fallback
            else:
                errors.append(f"Referenced diagnostic_logd file does not exist: {ref}")
                continue
        try:
            if resolved.stat().st_size <= 0:
                errors.append(f"Referenced diagnostic_logd file is empty: {ref}")
        except OSError as exc:
            errors.append(f"Could not stat diagnostic_logd file {ref}: {exc}")


def validate_report(path: Path, threshold: int = 0, verbose: bool = False) -> VerificationResult:
    errors: list[str] = []
    warnings: list[str] = []
    report_path = path.resolve()

    repo_root, git_warning = find_repo_root(report_path.parent)
    if verbose and git_warning:
        warnings.append(git_warning)

    report = read_json_report(report_path, errors)
    if report is None:
        return VerificationResult(str(report_path), False, threshold, errors=errors, warnings=warnings)
    if not isinstance(report, dict):
        errors.append("Diagnostic JSON root must be an object")
        return VerificationResult(str(report_path), False, threshold, errors=errors, warnings=warnings)

    validate_required_fields(report, errors)
    validate_generated_at(report.get("generated_at"), errors)

    commit = report.get("commit")
    if isinstance(commit, str) and not COMMIT_RE.fullmatch(commit):
        errors.append("Field 'commit' must be exactly 8 lowercase hex characters")

    if "chunk_size_bytes" in report:
        chunk_size = report.get("chunk_size_bytes")
        if chunk_size is not None and (not is_int(chunk_size) or chunk_size <= 0):
            errors.append("Field 'chunk_size_bytes' must be a positive integer or null")

    total, passed, failed = validate_modules(report, errors)
    validate_logd_paths(report, report_path, repo_root, errors, warnings)

    if passed is not None and passed < threshold:
        errors.append(f"Passing modules {passed} is below threshold {threshold}")

    modules = report.get("modules") if isinstance(report.get("modules"), list) else None
    return VerificationResult(
        path=str(report_path),
        ok=not errors,
        threshold=threshold,
        total_modules=total,
        passed=passed,
        failed=failed,
        errors=errors,
        warnings=warnings,
        modules=modules,
    )


def non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate build diagnostic JSON and referenced .logd artifacts."
    )
    parser.add_argument(
        "report",
        nargs="?",
        help="Path to diagnostic/build-<commit>.json. Defaults to the latest report under ./diagnostic.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print module details and non-fatal warnings")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    parser.add_argument(
        "--threshold",
        type=non_negative_int,
        default=0,
        help="Minimum number of passing modules required (default: 0)",
    )
    return parser.parse_args(argv)


def print_human(result: VerificationResult, verbose: bool = False) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"Diagnostic verification: {status}")
    print(f"Report: {result.path}")
    if result.total_modules is not None:
        print(
            f"Modules: {result.passed}/{result.total_modules} passed, "
            f"{result.failed} failed; threshold={result.threshold}"
        )

    for warning in result.warnings or []:
        print(f"WARNING: {warning}", file=sys.stderr)
    for error in result.errors or []:
        print(f"ERROR: {error}", file=sys.stderr)

    if verbose and result.modules:
        print("Module results:")
        for module in result.modules:
            name = module.get("name", "<unknown>")
            status = module.get("status", "<unknown>")
            elapsed = module.get("elapsed_seconds", "?")
            print(f"  - {name}: {status} ({elapsed}s)")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.report:
        report_path = Path(args.report)
    else:
        repo_root, _ = find_repo_root(Path.cwd())
        discovered = latest_diagnostic_report(repo_root)
        if discovered is None:
            result = VerificationResult(
                path=str(repo_root / "diagnostic"),
                ok=False,
                threshold=args.threshold,
                errors=["No diagnostic/build-*.json file found. Run `python3 build.py` first."],
                warnings=[],
            )
            if args.json:
                print(json.dumps(result.to_dict(verbose=args.verbose), indent=2))
            else:
                print_human(result, verbose=args.verbose)
            return 1
        report_path = discovered

    result = validate_report(report_path, threshold=args.threshold, verbose=args.verbose)
    if args.json:
        print(json.dumps(result.to_dict(verbose=args.verbose), indent=2))
    else:
        print_human(result, verbose=args.verbose)
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
