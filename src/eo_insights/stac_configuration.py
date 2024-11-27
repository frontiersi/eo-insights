"""Construct all available configurations"""

from pathlib import Path

from eo_insights.stac_utils import STACConfig
from eo_insights.utils import EOInsightsException

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR.joinpath("dataset_configuration")


# Manually create an instantiated STACConfig file for each file
# DIGITAL EARTH AFRICA
try:
    CONFIG_PATH = CONFIG_DIR.joinpath("digital_earth_africa_stac.toml")

    de_africa_stac_config = STACConfig.from_toml(
        configuration_toml_path=CONFIG_DIR.joinpath(CONFIG_PATH)
    )
except FileNotFoundError:
    raise EOInsightsException(
        f"The config file for Digital Earth Africa was not found. The expected location is: {CONFIG_PATH}"
    )

# DIGITAL EARTH AUSTRALIA
try:
    CONFIG_PATH = CONFIG_DIR.joinpath("digital_earth_australia_stac.toml")

    de_australia_stac_config = STACConfig.from_toml(
        configuration_toml_path=CONFIG_DIR.joinpath(CONFIG_PATH)
    )
except FileNotFoundError:
    raise EOInsightsException(
        f"The config file for Digital Earth Australia was not found. The expected location is: {CONFIG_PATH}"
    )

# ELEMENT 84
try:
    CONFIG_PATH = CONFIG_DIR.joinpath("element_84_stac.toml")

    element_84_stac_config = STACConfig.from_toml(
        configuration_toml_path=CONFIG_DIR.joinpath(CONFIG_PATH)
    )
except FileNotFoundError:
    raise EOInsightsException(
        f"The config file for Element 84 was not found. The expected location is: {CONFIG_PATH}"
    )

# NASA LP CLOUD
try:
    CONFIG_PATH = CONFIG_DIR.joinpath("nasa_lpcloud_stac.toml")

    nasa_lpcloud_stac_config = STACConfig.from_toml(
        configuration_toml_path=CONFIG_DIR.joinpath(CONFIG_PATH)
    )
except FileNotFoundError:
    raise EOInsightsException(
        f"The config file for NASA LPCLOUD was not found. The expected location is: {CONFIG_PATH}"
    )
