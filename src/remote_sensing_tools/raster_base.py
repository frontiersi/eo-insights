"""Data loading"""

import logging

from dataclasses import dataclass
from typing import Union, Optional

import odc.stac
import pystac_client
import xarray
from remote_sensing_tools.stac_utils import STACConfig, CollectionInfo
from remote_sensing_tools.masking import set_mask_attributes

# Construct types for type hinting
BBox = tuple[float, float, float, float]
XarrayType = Union[xarray.Dataset, xarray.DataArray]

# Set up the logger
_log = logging.getLogger(__name__)


@dataclass
class QueryParams:
    """
    Information required for querying a stac catalog
    """

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

    def __init__(
        self,
        data: Optional[XarrayType] = None,
        masks: Optional[XarrayType] = None,
    ):
        self.data = data
        self.masks = masks

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

        # Add masking attributes if a mask is present
        # Identify whether any of the masks are present in the loaded data
        requested_masks: list = []
        for requested_collection_name in collections:

            collection_info = config.collections.get(requested_collection_name, None)

            if collection_info is not None:
                configured_masks = collection_info.masks

                configured_mask_aliases = [
                    mask_info.alias for _, mask_info in configured_masks.items()
                ]

                matched_masks = set(configured_mask_aliases).intersection(data.keys())

                for mask in matched_masks:
                    data[mask] = set_mask_attributes(data[mask], configured_masks[mask])

                requested_masks.extend(matched_masks)

        # Ensure unique aliases before proceeding
        unique_masks = set(requested_masks)
        if len(unique_masks) != 0:
            masks = data[list(unique_masks)]
            data = data.drop_vars(list(unique_masks))
        else:
            masks = None

        return cls(data, masks)
