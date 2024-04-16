"""Data loading"""

from dataclasses import dataclass
from typing import TypeVar, Union
import pathlib

import odc.stac
import pystac_client
import xarray
from remote_sensing_tools.stac_config import STACConfig

# Construct types for type hinting
XarrayObject = Union[xarray.DataArray, xarray.Dataset]
XarrayTypeVar = TypeVar("XarrayTypeVar", xarray.DataArray, xarray.Dataset)
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


def open_stac_catalog(catalog_url: str) -> pystac_client.client.Client:
    """Connect to a stac catalog and return the client used to access"""

    # Connect to catalog
    catalog = pystac_client.Client.open(catalog_url)

    return catalog


def apply_rio_config(rio_config: dict) -> None:
    """Apply the rio config for odc-stac"""
    odc.stac.configure_rio(**rio_config)


def get_stac_items_from_query(
    catalog: pystac_client.client.Client,
    bbox: BBox,
    collections: list,
    start_date: str,
    end_date: str,
) -> list:
    """Return STAC items for a given query"""

    query = catalog.search(
        bbox=bbox,
        collections=collections,
        datetime=f"{start_date}/{end_date}",
    )

    # Search the STAC catalog for all items matching the query
    items = list(query.items())

    return items


def load_stac_items(
    items: list,
    bands: Union[tuple, list],
    crs: str,
    resolution: int,
    bbox: BBox,
    config: dict,
    chunks: dict,
) -> XarrayObject:
    """Load STAC items into an xarry datset or dataarray"""
    ds = odc.stac.load(
        items,
        bands=bands,
        crs=crs,
        resolution=resolution,
        groupby="solar_day",
        chunks=chunks,
        bbox=bbox,
        stac_cfg=config,
    )

    return ds


def apply_pq_mask(
    data: XarrayObject,
    masking_band: str,
    category_value_dictionary: dict,
    categories_to_mask: list,
    nodata_value: Union[float, int],
) -> None:
    """Apply the pixel quality mask"""

    # Separate out the masking band and the data to be masked
    mask_array = data[masking_band]
    data.drop_vars([masking_band])

    values_to_mask = [
        category_value_dictionary[category] for category in categories_to_mask
    ]

    inclusion_mask = ~mask_array.isin(values_to_mask)

    data.where(cond=inclusion_mask, other=nodata_value)


class RasterBase:
    """Class for instantiating raster data"""

    def __init__(self, data=None):
        self.data = data

    @classmethod
    def from_stac_query(
        cls,
        configuration_file: Union[str, pathlib.Path],
        product_code: str,
        query_params: QueryParams,
        load_params: LoadParams,
    ):
        """
        Specific factory method for building from stac query
        This returns a lazy-loaded xarray with masking applied, assuming
        there is a pq_mask band present
        """

        # Load data
        config = STACConfig(configuration_file)

        catalog = open_stac_catalog(config.catalog_url)

        apply_rio_config(config.rio_config)

        items = get_stac_items_from_query(
            catalog,
            bbox=query_params.bbox,
            collections=[product_code],
            start_date=query_params.start_date,
            end_date=query_params.end_date,
        )

        data = load_stac_items(
            items,
            bands=load_params.bands,
            crs=load_params.crs,
            resolution=load_params.resolution,
            bbox=query_params.bbox,
            config=config.config_dictionary,
            chunks={},
        )

        pq_mask_dictionary = config.get_pq_mask_dictionary(product_code)

        apply_pq_mask(
            data=data,
            masking_band="pq_mask",
            category_value_dictionary=pq_mask_dictionary["flags_definition"]["values"],
            categories_to_mask=pq_mask_dictionary["categories_to_mask"],
            nodata_value=config.get_product_dictionary(product_code)["assets"]["*"][
                "nodata"
            ],
        )

        return cls(data)
