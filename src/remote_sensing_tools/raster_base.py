"""Data loading"""

from dataclasses import dataclass
from typing import Union, Optional

import odc.stac
import pystac_client
import xarray
from remote_sensing_tools.stac_config import STACConfig

# Construct types for type hinting
BBox = tuple[float, float, float, float]


@dataclass
class QueryParams:
    """
    Information required for querying a stac catalog
    """

    product: str
    bbox: BBox
    start_date: str
    end_date: str


@dataclass
class LoadParams:
    """
    Information required to load from a stac catalog
    """

    crs: str
    resolution: int
    bands: Union[tuple, list]


class RasterBase:
    """Class for instantiating raster data"""

    def __init__(self, data: Optional[xarray.Dataset] = None):
        self.data = data

    @classmethod
    def from_stac_query(
        cls,
        config: STACConfig,
        collections: list[str],
        query_params: QueryParams,
        load_params: LoadParams,
    ):
        """
        Specific factory method for building from stac query
        This returns a lazy-loaded xarray
        """

        # Connect to a stac catalog and return the client used to access
        catalog = pystac_client.Client.open(config.catalog.url)

        # Apply the rio config for odc-stac
        odc.stac.configure_rio(**config.catalog.rio_config)

        # Run the STAC query
        query = catalog.search(
            bbox=query_params.bbox,
            collections=collections,
            datetime=f"{query_params.start_date}/{query_params.end_date}",
        )

        # List items returned by the query
        items = list(query.items())

        data = odc.stac.load(
            items,
            bands=load_params.bands,
            crs=load_params.crs,
            resolution=load_params.resolution,
            bbox=query_params.bbox,
            chunks={},
            groupby="solar_day",
            stac_cfg=config.configuration.get("collections", {}),
        )

        return cls(data)
