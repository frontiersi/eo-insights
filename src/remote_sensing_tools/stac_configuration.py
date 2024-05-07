"""Construct all available configurations"""

import logging

from pathlib import Path

from remote_sensing_tools.stac_utils import STACConfig

_log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = BASE_DIR.joinpath("dataset_configuration")

# Manually create an instantiated STACConfig file for each file
# DIGITAL EARTH AFRICA
try:
    CONFIG_PATH = CONFIG_DIR.joinpath("digital_earth_africa_stac.toml")

    de_africa_stac_config = STACConfig.from_toml(
        configuration_toml_path=CONFIG_DIR.joinpath(CONFIG_PATH)
    )
except FileNotFoundError:
    _log.exception(
        "The config file for Digital Earth Africa was not found. The expected location is: %s",
        CONFIG_PATH,
    )

# DIGITAL EARTH AUSTRALIA
try:
    CONFIG_PATH = CONFIG_DIR.joinpath("digital_earth_australia_stac.toml")

    de_australia_stac_config = STACConfig.from_toml(
        configuration_toml_path=CONFIG_DIR.joinpath(CONFIG_PATH)
    )
except FileNotFoundError:
    _log.exception(
        "The config file for Digital Earth Australia was not found. The expected location is: %s",
        CONFIG_PATH,
    )

try:
    CONFIG_PATH = CONFIG_DIR.joinpath("element_84_stac.toml")

    element_84_stac_config = STACConfig.from_toml(
        configuration_toml_path=CONFIG_DIR.joinpath(CONFIG_PATH)
    )
except FileNotFoundError:
    _log.exception(
        "The config file for Digital Earth Australia was not found. The expected location is: %s",
        CONFIG_PATH,
    )
