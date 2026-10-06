"""Generate atlas and documentation variable specification JSON files."""

import json
from pathlib import Path

from tidal_fvcom.variable_registry import (
    atlas_variable_specification,
    documentation_variable_specification,
)

SCRIPT_DIR = Path(__file__).resolve().parent
ATLAS_SPEC = SCRIPT_DIR.parents[1] / "atlas" / "atlas_variable_spec.json"
DOC_SPEC = SCRIPT_DIR / "documentation_variable_spec.json"

with open(ATLAS_SPEC, "w") as f:
    json.dump(atlas_variable_specification, f, indent=2)

with open(DOC_SPEC, "w") as f:
    json.dump(documentation_variable_specification, f, indent=2)

print(f"Wrote atlas_variable_spec.json ({len(atlas_variable_specification)} vars)")
print(f"Wrote documentation_variable_spec.json ({len(documentation_variable_specification)} vars)")
