import numpy as np
import pandas as pd
import pytest

from tidal_fvcom.atlas_element_filter import (
    apply_me_atlas_element_filter,
    build_element_mask,
)
from tidal_fvcom.config import config

SPEED = "vap_water_column_mean_sea_water_speed"


def se_ak_location():
    return config["location_specification"]["southeast_alaska"]


def make_df():
    # face_id: what each face should do
    # 0 inside box, high land, zero speed -> removed
    # 1 inside box, high land, moving     -> kept
    # 2 inside box, shallow land, zero    -> kept
    # 3 outside box, high land, zero      -> kept
    # 4 on the box edge, high land, zero  -> removed (edges inclusive)
    # 5 inside box, deep water, zero      -> kept
    return pd.DataFrame(
        {
            "face_id": [0, 1, 2, 3, 4, 5],
            "lon_center": [-130.8, -130.8, -130.8, -129.0, -131.15, -130.8],
            "lat_center": [54.3, 54.3, 54.3, 54.3, 54.7, 54.3],
            "h_center": [-50.0, -50.0, -5.0, -50.0, -50.0, 80.0],
            SPEED: [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        },
        index=[10, 11, 12, 13, 14, 15],
    )


def website_island_fix_mask(df, settings):
    # Copy of island_fix_mask from resource_geoparquet_to_tiles
    mask = (df["h_center"] <= 0).values & (df["h_center"] < settings["hMax"]).values
    if settings.get("bbox"):
        west, south, east, north = settings["bbox"]
        mask &= (
            df["lon_center"].between(west, east) & df["lat_center"].between(south, north)
        ).values
    if settings.get("speedMax") is not None:
        mask &= (df[SPEED] <= settings["speedMax"]).values
    return mask


def test_removes_only_faces_matching_all_rules():
    df = make_df()
    out = apply_me_atlas_element_filter(df, se_ak_location())
    assert out["face_id"].tolist() == [1, 2, 3, 5]


def test_row_order_and_index_preserved():
    df = make_df()
    out = apply_me_atlas_element_filter(df, se_ak_location())
    assert out.index.tolist() == [11, 12, 13, 15]


def test_matches_website_rule():
    df = make_df()
    spec = se_ak_location()["me_atlas_element_filter"]
    settings = {"bbox": [-131.15, 54.05, -130.4, 54.7], "hMax": -10.0, "speedMax": 0.0}
    assert np.array_equal(build_element_mask(df, spec), website_island_fix_mask(df, settings))


def test_location_without_filter_is_unchanged():
    df = make_df()
    out = apply_me_atlas_element_filter(df, {"output_name": "AK_cook_inlet"})
    assert out is df


def test_disabled_rule_is_skipped():
    df = make_df()
    spec = {
        "rules": [
            {"type": "column", "enabled": False, "column": SPEED, "op": "<=", "value": 0.0},
            {"type": "column", "column": "h_center", "op": "<", "value": -10.0},
        ]
    }
    # Only the h_center rule is active: faces 0, 1, 3, 4
    assert np.flatnonzero(build_element_mask(df, spec)).tolist() == [0, 1, 3, 4]


def test_all_rules_disabled_removes_nothing():
    df = make_df()
    spec = {"rules": [{"type": "column", "enabled": False, "column": "h_center", "op": "<", "value": 0}]}
    assert not build_element_mask(df, spec).any()


def test_bad_operator_raises():
    spec = {"rules": [{"type": "column", "column": "h_center", "op": "~", "value": 0}]}
    with pytest.raises(ValueError):
        build_element_mask(make_df(), spec)


def test_unknown_rule_type_raises():
    with pytest.raises(ValueError):
        build_element_mask(make_df(), {"rules": [{"type": "polygon"}]})


def test_missing_column_raises():
    spec = {"rules": [{"type": "column", "column": "nope", "op": "<", "value": 0}]}
    with pytest.raises(KeyError):
        build_element_mask(make_df(), spec)


def test_se_ak_validator_rejects_incomplete_spec():
    location = {
        "output_name": "AK_southeast",
        "me_atlas_element_filter": {"rules": [{"type": "column", "column": "h_center", "op": "<", "value": -10}]},
    }
    with pytest.raises(ValueError):
        apply_me_atlas_element_filter(make_df(), location)
