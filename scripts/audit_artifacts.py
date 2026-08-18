from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SPEC_OUTPUT_BUCKETS = {
    "paper_main",
    "reviewer_suite",
    "ablations",
    "stress_tests",
    "debug",
    "cache",
    "failed",
    "figures",
    "tables",
    "reports",
}
REQUIRED_METADATA_FIELDS = (
    "run_id",
    "git_commit",
    "created_at",
    "config_path",
    "config_hash",
    "dataset",
    "attack_type",
    "target_model",
    "variant",
    "seed",
    "stage",
    "rq",
    "status",
    "failure_reason",
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _coerce_metadata(payload: dict[str, Any], path: Path) -> tuple[dict[str, Any], str]:
    if path.name == "manifest.json":
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            return metadata, "manifest"
        return {}, "manifest"
    return payload, "run_metadata"


def _missing_metadata_fields(metadata: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for key in REQUIRED_METADATA_FIELDS:
        if key not in metadata:
            missing.append(key)
    return missing


def _empty_required_metadata_fields(metadata: dict[str, Any]) -> list[str]:
    optional_empty = {"git_commit", "attack_type", "variant", "failure_reason"}
    empty: list[str] = []
    for key in REQUIRED_METADATA_FIELDS:
        if key in optional_empty:
            continue
        value = metadata.get(key)
        if value is None or str(value).strip() == "":
            empty.append(key)
    return empty


def _check_manifest_outputs(path: Path, payload: dict[str, Any]) -> list[str]:
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict):
        return []
    missing_files: list[str] = []
    stage_name = str(payload.get("stage") or payload.get("metadata", {}).get("stage") or "").strip()
    for output_key, rel_name in outputs.items():
        text = str(rel_name or "").strip()
        if not text:
            continue
        target = path.parent / text
        if not target.exists():
            if _is_volatile_cache_output(stage_name, str(output_key), target):
                continue
            missing_files.append(text)
    return missing_files


def _is_volatile_cache_output(stage_name: str, output_key: str, path: Path) -> bool:
    """Generated caches are reproducible helpers, not durable evidence artifacts."""
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path.absolute()
    if ".cache" in resolved.parts:
        return True
    return stage_name == "data_prep" and output_key in {"data_cache", "data_artifact_dir"}


def audit_outputs(outputs_root: Path) -> dict[str, Any]:
    outputs_root = outputs_root.resolve()
    issues: list[dict[str, str]] = []
    manifest_count = 0
    run_metadata_count = 0
    failed_record_count = 0

    if not outputs_root.exists():
        issues.append(
            {
                "severity": "error",
                "kind": "missing_outputs_root",
                "path": str(outputs_root).replace("\\", "/"),
                "message": "outputs root does not exist",
            }
        )
        return {
            "status": "fail",
            "outputs_root": str(outputs_root).replace("\\", "/"),
            "issues": issues,
            "counts": {"errors": 1, "warnings": 0, "manifests": 0, "run_metadata": 0, "failed_records": 0},
        }

    for child in sorted(path for path in outputs_root.iterdir() if path.is_dir()):
        if child.name not in SPEC_OUTPUT_BUCKETS:
            issues.append(
                {
                    "severity": "warning",
                    "kind": "non_spec_output_bucket",
                    "path": _relpath(child, outputs_root),
                    "message": f"top-level output bucket '{child.name}' is not declared in SPEC.md",
                }
            )

    audit_targets = sorted(outputs_root.rglob("manifest.json")) + sorted(outputs_root.rglob("run_metadata.json"))
    for path in audit_targets:
        payload = _load_json(path)
        metadata, audit_type = _coerce_metadata(payload, path)
        if audit_type == "manifest":
            manifest_count += 1
        else:
            run_metadata_count += 1
        if not metadata:
            issues.append(
                {
                    "severity": "error",
                    "kind": "missing_metadata",
                    "path": _relpath(path, outputs_root),
                    "message": f"{audit_type} is missing structured metadata",
                }
            )
            continue

        missing = _missing_metadata_fields(metadata)
        if missing:
            issues.append(
                {
                    "severity": "error",
                    "kind": "missing_metadata_fields",
                    "path": _relpath(path, outputs_root),
                    "message": f"missing metadata fields: {', '.join(missing)}",
                }
            )
        empty = _empty_required_metadata_fields(metadata)
        if empty:
            issues.append(
                {
                    "severity": "warning",
                    "kind": "empty_metadata_fields",
                    "path": _relpath(path, outputs_root),
                    "message": f"empty required metadata fields: {', '.join(empty)}",
                }
            )

        if path.name == "manifest.json":
            missing_outputs = _check_manifest_outputs(path, payload)
            if missing_outputs:
                issues.append(
                    {
                        "severity": "error",
                        "kind": "missing_declared_outputs",
                        "path": _relpath(path, outputs_root),
                        "message": f"declared outputs missing on disk: {', '.join(missing_outputs)}",
                    }
                )

        if str(metadata.get("status", "")).strip().lower() == "failed":
            failed_record_count += 1

    failed_root = outputs_root / "failed"
    if failed_root.exists():
        for path in sorted(failed_root.rglob("*.json")):
            payload = _load_json(path)
            metadata = payload.get("metadata")
            if not isinstance(metadata, dict):
                issues.append(
                    {
                        "severity": "error",
                        "kind": "invalid_failed_record",
                        "path": _relpath(path, outputs_root),
                        "message": "failed record is missing metadata",
                    }
                )
                continue
            failed_record_count += 1
            status = str(metadata.get("status", "")).strip().lower()
            if status != "failed":
                issues.append(
                    {
                        "severity": "error",
                        "kind": "failed_record_status_mismatch",
                        "path": _relpath(path, outputs_root),
                        "message": f"failed record metadata.status must be 'failed', got '{status}'",
                    }
                )

    error_count = sum(1 for issue in issues if issue["severity"] == "error")
    warning_count = sum(1 for issue in issues if issue["severity"] == "warning")
    status = "fail" if error_count else ("warn" if warning_count else "ok")
    return {
        "status": status,
        "outputs_root": str(outputs_root).replace("\\", "/"),
        "issues": issues,
        "counts": {
            "errors": error_count,
            "warnings": warning_count,
            "manifests": manifest_count,
            "run_metadata": run_metadata_count,
            "failed_records": failed_record_count,
        },
    }


def _render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Artifact Audit Report",
        "",
        f"- outputs_root: `{report['outputs_root']}`",
        f"- status: `{report['status']}`",
        f"- manifests: `{report['counts']['manifests']}`",
        f"- run_metadata: `{report['counts']['run_metadata']}`",
        f"- failed_records: `{report['counts']['failed_records']}`",
        f"- errors: `{report['counts']['errors']}`",
        f"- warnings: `{report['counts']['warnings']}`",
        "",
    ]
    if not report["issues"]:
        lines.append("No audit issues detected.")
        return "\n".join(lines) + "\n"

    lines.extend(["## Issues", ""])
    for issue in report["issues"]:
        lines.append(
            f"- [{issue['severity'].upper()}] `{issue['kind']}` at `{issue['path']}`: {issue['message']}"
        )
    lines.append("")
    return "\n".join(lines)


def write_audit_report(outputs_root: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    reports_dir = outputs_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "artifact_audit_report.json"
    md_path = reports_dir / "artifact_audit_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown_report(report), encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit output artifacts against SPEC metadata and layout rules.")
    parser.add_argument("--root", default="outputs", help="Outputs root directory to audit.")
    args = parser.parse_args()

    outputs_root = (ROOT / args.root).resolve() if not Path(args.root).is_absolute() else Path(args.root).resolve()
    report = audit_outputs(outputs_root)
    json_path, md_path = write_audit_report(outputs_root, report)

    print(f"[artifact-audit] status={report['status']}")
    print(f"[artifact-audit] outputs_root={report['outputs_root']}")
    print(
        "[artifact-audit] counts"
        f" manifests={report['counts']['manifests']}"
        f" run_metadata={report['counts']['run_metadata']}"
        f" failed_records={report['counts']['failed_records']}"
        f" errors={report['counts']['errors']}"
        f" warnings={report['counts']['warnings']}"
    )
    print(f"[artifact-audit] json={json_path}")
    print(f"[artifact-audit] markdown={md_path}")
    for issue in report["issues"][:20]:
        print(f"[artifact-audit][{issue['severity'].upper()}] {issue['kind']} {issue['path']} :: {issue['message']}")

    if report["status"] == "fail":
        return 2
    if report["status"] == "warn":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
