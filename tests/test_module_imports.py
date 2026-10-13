"""The package modules import and the config loads in the pixi env."""

import importlib

import pytest

CORE_MODULES = [
    "tidal_fvcom",
    "tidal_fvcom.config",
    "tidal_fvcom.standardize",
    "tidal_fvcom.derive_vap_fvcom",
    "tidal_fvcom.nc_manager",
    "tidal_fvcom.time_manager",
    "tidal_fvcom.coord_manager",
    "tidal_fvcom.attrs_manager",
    "tidal_fvcom.file_manager",
    "tidal_fvcom.verify",
]


@pytest.mark.parametrize("name", CORE_MODULES)
def test_core_module_imports(name):
    importlib.import_module(name)


def test_config_has_locations():
    from tidal_fvcom.config import config

    locations = config["location_specification"]
    assert "cook_inlet" in locations
