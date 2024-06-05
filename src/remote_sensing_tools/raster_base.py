"""Data loading"""

import logging

from dataclasses import dataclass
from typing import Union, Optional

import odc.stac
import pystac_client
import xarray
from remote_sensing_tools.stac_utils import STACConfig
from remote_sensing_tools.masking import set_mask_attributes, convert_mask_to_bool

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
        data: Optional[xarray.Dataset] = None,
        masks: Optional[xarray.Dataset] = None,
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

    def generate_boolean_mask(self, mask_name: str, inplace: bool = True):
        """
        For a given mask, create the relevant boolean mask

        Parameters
        ----------
        mask_name : str
            Name of the mask.
        inplace : bool, optional
            Whether to modify the mask inplace, by default True.
            If False, a new variable will be created with mask_name followed by _bool.
        """

        if self.masks is None:
            _log.warning(
                "The mask attribute for this %s is None. No mask can be generated",
                type(self),
            )
        else:
            boolean_mask = convert_mask_to_bool(
                masks_ds=self.masks, mask_name=mask_name
            )

            # If the user is doing this inplace, the boolean mask will overwrite the original mask
            if inplace:
                destination = mask_name
            else:
                destination = f"{mask_name}_bool"

            self.masks[destination] = boolean_mask

    def apply_mask(
        self, mask_name: str, data_inplace: bool = True, mask_inplace: bool = True
    ):
        """
        For a given mask, if it's a boolean, apply to data.
        If not, create the boolean mask using generate_boolean_mask, then apply.

        Parameters
        ----------
        mask_name : str
            _description_
        data_inplace : bool, optional
            _description_, by default True
        mask_inplace : bool, optional
            _description_, by default True

        Raises
        ------
        ValueError
            If `self.data` is `None`
        ValueError
            If `self.masks` is `None`
        KeyError
            If `self.masks` has no data variable `mask_name`

        """
        if self.data is None:
            raise ValueError(
                f"The data attribute for this {type(self)} is None."
                "There is no data to apply the mask to."
            )

        if self.masks is None:
            raise ValueError(
                f"The mask attribute for this {type(self)} is None."
                "There is no mask to be applied."
            )
        else:
            try:
                mask = self.masks[mask_name]
            except KeyError as e:
                raise KeyError(
                    f"Mask '{mask_name}' was not recognised."
                    f"Available masks are {list(self.masks.data_vars)}."
                ) from e

        # If not dealing with a boolean mask, create the boolean version
        if mask.attrs.get("mask_type") != "boolean":
            self.generate_boolean_mask(mask_name, inplace=mask_inplace)
            if mask_inplace is False:
                mask = self.masks[f"{mask_name}_bool"]

        # Invert the mask, which is True for bad values, False for good values by design
        # Inverting allows us to apply it and keep the good values.
        inverted_mask = ~mask

        # Identify unmasked variables -- this is to ward against mutliple re-runs with inplace=False
        unmasked_vars = [
            var for var in list(self.data.data_vars) if "_masked" not in var
        ]

        # Apply the mask to each variable
        for variable in unmasked_vars:
            if data_inplace:
                destination = variable
            else:
                destination = f"{variable}_masked"

            self.data[destination] = self.data[variable].where(
                inverted_mask, other=self.data[variable].odc.nodata
            )
