"""One-command variable documentation regenerator.

Runs the full pipeline: registry -> JSON -> MD files.

Usage:
    python scripts/docs_gen/generate_variable_docs.py [--pages-dir DIR]
"""

import argparse
import json
from pathlib import Path

from tidal_fvcom.variable_registry import (
    atlas_variable_specification,
    documentation_variable_specification,
)
from generate_mkdocs_variable_section import generate_all_pages, PAGES_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pages-dir",
        type=Path,
        default=PAGES_DIR,
        help=f"Where to write variable MD pages (default: {PAGES_DIR})",
    )
    pages_dir = parser.parse_args().pages_dir

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]

    # Step 1: Write JSON spec files from registry
    atlas_path = repo_root / "atlas" / "atlas_variable_spec.json"
    doc_path = script_dir / "documentation_variable_spec.json"

    atlas_path.write_text(json.dumps(atlas_variable_specification, indent=2) + "\n")
    print(f"Wrote {atlas_path.name} ({len(atlas_variable_specification)} vars)")

    doc_path.write_text(json.dumps(documentation_variable_specification, indent=2) + "\n")
    print(f"Wrote {doc_path.name} ({len(documentation_variable_specification)} vars)")

    # Step 2: Generate and write MD pages from the doc spec
    pages = generate_all_pages(documentation_variable_specification)

    pages_dir.mkdir(parents=True, exist_ok=True)

    changed = 0
    unchanged = 0
    for filename, content in pages.items():
        page_path = pages_dir / filename
        if page_path.exists() and page_path.read_text() == content:
            unchanged += 1
        else:
            page_path.write_text(content)
            print(f"  Updated {page_path}")
            changed += 1

    print(f"\nDone: {changed} files updated, {unchanged} unchanged.")


if __name__ == "__main__":
    main()
