"""Data loading"""

from dataclasses import dataclass

import odc.stac
import pystac_client


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

    def load_from_stac(self, catalog_url, collections, stac_config, aws_endpoint):
        """Method to load from STAC catalog"""

        # Connect to catalog
        catalog = pystac_client.Client.open(catalog_url)

        # Configure rasterio to connect to AWS. This is required for DE Africa
        # but may not be required for other catalogs
        odc.stac.configure_rio(
            cloud_defaults=True,
            aws={"aws_unsigned": True},
            AWS_S3_ENDPOINT=aws_endpoint,
        )

        # Search the catalog using the bounding box and start/end dates from the load info
        query = catalog.search(
            bbox=self.load_info.bbox,
            collections=collections,
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
            stac_cfg=stac_config,
        )

        return ds
