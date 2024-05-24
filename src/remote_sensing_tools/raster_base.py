"""Data loading"""

import logging

from dataclasses import dataclass
from typing import Union, Optional

import odc.stac
import pystac_client
import xarray
from remote_sensing_tools.stac_utils import STACConfig
from remote_sensing_tools.masking import set_mask_attributes, generate_categorical_mask

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

            configured_masks = config.get_collection_masks(requested_collection_name)

            if len(configured_masks) > 0:

                matched_masks = set(configured_masks.keys()).intersection(data.keys())

                for mask in matched_masks:
                    data[mask] = set_mask_attributes(data[mask], configured_masks[mask])

                requested_masks.extend(matched_masks)

            else:
                _log.warning(
                    "There were no configured masks found for %s",
                    requested_collection_name,
                )

        # Ensure unique aliases before proceeding
        unique_masks = set(requested_masks)
        if len(unique_masks) != 0:
            masks = data[list(unique_masks)]
            data = data.drop_vars(list(unique_masks))
        else:
            masks = None
            _log.info("No masks were found for the requested collections")

        return cls(data, masks)

    def apply_mask(self, mask_name):
        """For a given mask, create the relevant boolean mask and apply to data"""

        try:
            mask = self.masks[mask_name]
        except KeyError as e:
            raise KeyError(
                f"Mask '{mask_name}' was not recognised. Available masks are {list(self.masks.data_vars)}"
            ) from e

        # Determine the mask type from the mask attributes
        mask_type = mask.attrs.get("mask_type")

        if mask_type is not None:
            if mask_type == "categorical":
                mask_categories = mask.attrs.get("categories_to_mask")
                mask_category_values = mask.attrs.get("flags_definition").get("values")

                _log.info(
                    "Selecting all pixels belonging to any of %s", mask_categories
                )
                self.masks[mask_name] = generate_categorical_mask(
                    mask=mask,
                    categories=mask_categories,
                    category_values=mask_category_values,
                )
            else:
                raise NotImplementedError(
                    f"No mask generation functionality exists for masks of type {mask_type}. Valid mask types are ['categorical']"
                )
        else:
            raise ValueError(
                "No mask type was found. Ensure the mask has a valid 'mask_type' attribute."
            )

        # Invert the boolean mask before applying it to the data
        inverted_mask = ~self.masks[mask_name]

        # Apply the mask to each variable
        for variable in self.data:
            self.data[variable] = self.data[variable].where(
                inverted_mask, other=self.data[variable].odc.nodata
            )
