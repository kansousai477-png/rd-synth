from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rdsynth.plotting.reviewer_figures import PlotTheme, generate_reviewer_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reviewer-facing figures from existing reviewer-suite artifacts.")
    parser.add_argument("--root", required=True, help="Reviewer-suite run root, e.g. outputs/reviewer_suite/runs/<run_id>.")
    parser.add_argument("--dataset", required=True, help="Dataset key under the run root, e.g. unsw.")
    parser.add_argument("--out-dir", default=None, help="Optional figure output directory. Defaults to <root>/<dataset>/figures.")
    parser.add_argument("--theme", default="paper", help="Built-in theme name: paper, mono, warm.")
    parser.add_argument("--theme-json", default=None, help="Optional JSON file that overrides theme fields.")
    args = parser.parse_args()

    run_root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else None
    theme_json = Path(args.theme_json).resolve() if args.theme_json else None
    theme = PlotTheme.from_sources(args.theme, theme_json)
    artifacts = generate_reviewer_figures(
        run_root=run_root,
        dataset=args.dataset,
        out_dir=out_dir,
        theme=theme,
    )
    for artifact in artifacts:
        print(f"[Figure] {artifact.key}: {artifact.png_path}")
    print(f"[Figure] bank: {run_root / f'{args.dataset.upper()}_FIGURE_BANK_CN.md'}")


if __name__ == "__main__":
    main()
