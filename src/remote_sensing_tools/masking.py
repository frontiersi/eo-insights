"""Functionality for managing data masks"""

import logging

from typing import Union, Optional
import xarray
from remote_sensing_tools.stac_config import MaskInfo

XarrayType = Union[xarray.Dataset, xarray.DataArray]

# Set up the logger
_log = logging.getLogger(__name__)


def set_mask_attributes(
    mask: xarray.DataArray, mask_info: Optional[MaskInfo]
) -> xarray.DataArray:
    """Attach information in MaskInfo to xarray.DataArray attribute"""

    if (mask_info is not None) and (mask.name == mask_info.alias):
        mask.attrs.update(
            collection=mask_info.collection,
            type=mask_info.type,
            categories_to_mask=mask_info.categories_to_mask,
            flags_definition=mask_info.flags_definition,
        )
    else:
        _log.error(
            "Mask band %s did not match provided MaskInfo %s",
            mask.name,
            mask_info.alias,
        )

    return mask
