"""Functionality for managing data masks"""

import logging
from typing import Union
import xarray
from remote_sensing_tools.stac_utils import MaskInfo

XarrayType = Union[xarray.Dataset, xarray.DataArray]

_log = logging.getLogger(__name__)


def set_mask_attributes(
    mask: xarray.DataArray, mask_info: MaskInfo
) -> xarray.DataArray:
    """Attach information in MaskInfo to xarray.DataArray attribute"""

    if mask.name == mask_info.alias:
        mask.attrs.update(
            collection=mask_info.collection,
            mask_type=mask_info.mask_type,
            categories_to_mask=mask_info.categories_to_mask,
            flags_definition=mask_info.flags_definition,
        )
    else:
        raise ValueError(
            f"Mask ({mask.name}) did not match MaskInfo ({mask_info.alias}). No attributes were added to {mask.name}"
        )

    return mask


def generate_categorical_mask(
    mask: xarray.DataArray, categories: list[str], category_values: dict[str, int]
) -> xarray.DataArray:
    """Generate a categorical mask"""

    values = [category_values[category] for category in categories]

    mask_bool = mask.isin(values)

    mask_bool.attrs = mask.attrs
    mask_bool.attrs.update(mask_type="boolean")

    return mask_bool
