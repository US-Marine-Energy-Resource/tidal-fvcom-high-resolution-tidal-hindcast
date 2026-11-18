"""Remove unwanted elements (faces) from the b5 atlas output.

The filter is described in config.py under a location, key
"me_atlas_element_filter". It is only used for the b5 atlas data. The b4
summary and the combined outputs are not changed.

Example spec:

    {
        "mode": "remove_if_all",
        "rules": [
            {"type": "bbox", "enabled": True, "west": -131.15, "south": 54.05,
             "east": -130.4, "north": 54.7},
            {"type": "column", "enabled": True, "column": "h_center",
             "op": "<", "value": -10.0},
        ],
    }

A face is removed when it matches all enabled rules. Rules that are not enabled
are skipped. If no rule is enabled, no face is removed.
"""

import operator

import numpy as np

CONFIG_KEY = "me_atlas_element_filter"

SUPPORTED_MODES = ("remove_if_all",)

OPERATORS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}

BBOX_KEYS = ("west", "south", "east", "north")


def _require_columns(df, columns, rule):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Filter rule {rule} needs missing column(s): {missing}")


def _bbox_mask(df, rule):
    """Faces with a center point inside the box. Edges are inclusive."""
    missing_keys = [k for k in BBOX_KEYS if k not in rule]
    if missing_keys:
        raise ValueError(f"bbox rule {rule} is missing keys: {missing_keys}")
    lon_col = rule.get("lon_column", "lon_center")
    lat_col = rule.get("lat_column", "lat_center")
    _require_columns(df, [lon_col, lat_col], rule)
    in_lon = df[lon_col].between(rule["west"], rule["east"])
    in_lat = df[lat_col].between(rule["south"], rule["north"])
    return (in_lon & in_lat).to_numpy()


def _column_mask(df, rule):
    """Faces where column <op> value is true. NaN never matches."""
    for key in ("column", "op", "value"):
        if key not in rule:
            raise ValueError(f"column rule {rule} is missing key: {key}")
    if rule["op"] not in OPERATORS:
        raise ValueError(
            f"Unknown operator {rule['op']!r}. Use one of {list(OPERATORS)}"
        )
    _require_columns(df, [rule["column"]], rule)
    return OPERATORS[rule["op"]](df[rule["column"]], rule["value"]).to_numpy()


RULE_TYPES = {
    "bbox": _bbox_mask,
    "column": _column_mask,
}


def build_element_mask(df, filter_spec):
    """Return a bool array, True for faces to remove."""
    mode = filter_spec.get("mode", "remove_if_all")
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unknown filter mode {mode!r}. Use one of {SUPPORTED_MODES}")

    enabled_rules = [r for r in filter_spec.get("rules", []) if r.get("enabled", True)]
    if not enabled_rules:
        return np.zeros(len(df), dtype=bool)

    mask = np.ones(len(df), dtype=bool)
    for rule in enabled_rules:
        rule_type = rule.get("type")
        if rule_type not in RULE_TYPES:
            raise ValueError(
                f"Unknown rule type {rule_type!r}. Use one of {list(RULE_TYPES)}"
            )
        mask &= RULE_TYPES[rule_type](df, rule)
    return mask


def validate_southeast_alaska_island_filter(filter_spec):
    """Check the SE_AK filter has the three island rules: area, h_center, speed."""
    rules = filter_spec.get("rules", [])
    bbox_rules = [r for r in rules if r.get("type") == "bbox"]
    column_names = {r.get("column") for r in rules if r.get("type") == "column"}

    if len(bbox_rules) != 1:
        raise ValueError("SE_AK island filter needs exactly one bbox rule")
    required = {"h_center", "vap_water_column_mean_sea_water_speed"}
    if not required.issubset(column_names):
        raise ValueError(f"SE_AK island filter needs column rules for {sorted(required)}")


# Location specific checks, keyed by the location output_name
LOCATION_VALIDATORS = {
    "AK_southeast": validate_southeast_alaska_island_filter,
}


def apply_me_atlas_element_filter(df, location):
    """Drop filtered faces from df using the location filter spec.

    Return df unchanged if the location has no filter. The row order and the
    face_id values of the kept rows are not changed.
    """
    filter_spec = location.get(CONFIG_KEY)
    if not filter_spec:
        return df

    validator = LOCATION_VALIDATORS.get(location.get("output_name"))
    if validator is not None:
        validator(filter_spec)

    remove = build_element_mask(df, filter_spec)
    print(
        f"Atlas element filter ({location.get('output_name')}): "
        f"removed {int(remove.sum())} of {len(df)} faces"
    )
    return df.loc[~remove]

