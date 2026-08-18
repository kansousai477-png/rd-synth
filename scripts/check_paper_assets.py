from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pypdf


def _extract_head_text(pdf_path: Path, max_pages: int = 3) -> str:
    reader = pypdf.PdfReader(str(pdf_path))
    pages = min(max_pages, len(reader.pages))
    text = []
    for i in range(pages):
        text.append((reader.pages[i].extract_text() or "").lower())
    return " ".join(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate baseline paper PDF assets against expected keywords.")
    parser.add_argument(
        "--manifest",
        default="paper/baselines/attack/paper_asset_manifest.json",
        help="Path to the paper asset manifest JSON.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    manifest_path = root / args.manifest
    if not manifest_path.exists():
        print(f"[paper-assets] manifest not found: {manifest_path}")
        return 2

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = payload.get("assets", [])
    failed = False
    for asset in assets:
        name = str(asset.get("name", "unknown"))
        path = root / str(asset.get("path", ""))
        keywords = [str(k).strip().lower() for k in asset.get("keywords_any", []) if str(k).strip()]
        if not path.exists():
            print(f"[paper-assets][FAIL] {name}: missing file {path}")
            failed = True
            continue
        try:
            head = _extract_head_text(path)
        except Exception as exc:
            print(f"[paper-assets][FAIL] {name}: unable to parse PDF ({exc})")
            failed = True
            continue
        if keywords and not any(keyword in head for keyword in keywords):
            print(f"[paper-assets][FAIL] {name}: keywords not found in head pages -> {keywords}")
            failed = True
        else:
            print(f"[paper-assets][OK] {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
