"""Data loading"""

from dataclasses import dataclass

import odc.stac
import pystac_client
import tomli


@dataclass
class LoadInfo:
    """
    Information required for loading
    """

    bbox: tuple
    crs: str
    bands: list
    resolution: int
    start_date: str
    end_date: str


class EOData:
    """EOData class"""

    def __init__(self, load_info: LoadInfo) -> None:
        self.load_info = load_info

    def load_from_stac(self, config_file, product):
        """Method to load from STAC catalog"""

        # Load the configuration.toml file
        with open(config_file, mode="rb") as f:
            config = tomli.load(f)

        # Connect to catalog
        catalog = pystac_client.Client.open(config["stac"]["catalog_url"])

        # Configure rasterio to connect to AWS. This is required for DE Africa
        # but may not be required for other catalogs
        odc.stac.configure_rio(**config["stac"]["rio_config"])

        # Search the catalog using the bounding box and start/end dates from the load info
        query = catalog.search(
            bbox=self.load_info.bbox,
            collections=[config["stac"]["configured_products"][product]],
            datetime=f"{self.load_info.start_date}/{self.load_info.end_date}",
        )

        # Search the STAC catalog for all items matching the query
        items = list(query.items())

        # Perform the stac load
        ds = odc.stac.load(
            items,
            bands=self.load_info.bands,
            crs=self.load_info.crs,
            resolution=self.load_info.resolution,
            groupby="solar_day",
            chunks={},
            bbox=self.load_info.bbox,
            stac_cfg=config,
        )

        return ds
