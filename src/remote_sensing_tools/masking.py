"""Functionality for managing data masks"""

import logging
from typing import Union, Literal
import xarray
from remote_sensing_tools.stac_utils import MaskInfo

XarrayType = Union[xarray.Dataset, xarray.DataArray]
MorphOperation = Literal["dilation", "erosion", "opening", "closing"]

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


def convert_mask_to_bool(masks_ds: xarray.Dataset, mask_name: str) -> xarray.DataArray:
    """Convert a single mask_name from masks_ds from its base type to a boolean"""

    try:
        mask = masks_ds[mask_name]
    except KeyError as e:
        raise KeyError(
            f"Mask band '{mask_name}' was not recognised."
            f"Available masks are {list(masks_ds.data_vars)}"
        ) from e

    # Determine the mask type from the mask attributes
    mask_type = mask.attrs.get("mask_type")

    if mask_type is not None:

        # Convert categorical mask to boolean
        if mask_type == "categorical":
            # Get categories from metadata
            mask_categories = mask.attrs.get("categories_to_mask")
            if mask_categories is None:
                raise ValueError(
                    f"Mask band {mask.name} has no categories to mask."
                    "Check metadata for categories to mask."
                )

            # Get flags definition values from metadata
            mask_flags_definition = mask.attrs.get("flags_definition")
            if mask_flags_definition is not None:
                mask_category_values = mask_flags_definition.get("values")
            else:
                raise ValueError(
                    f"Mask band {mask.name} has no flag definitions. Check metadata for mask."
                )

            _log.info("Converting categorical mask to boolean")
            _log.info("Selecting all pixels belonging to any of %s", mask_categories)

            masks_ds_bool = generate_categorical_mask(
                mask=mask,
                categories=mask_categories,
                category_values=mask_category_values,
            )
        elif mask_type == "boolean":
            _log.info("Using boolean mask as is.")
            masks_ds_bool = mask
        else:
            raise NotImplementedError(
                f"Mask band {mask.name} has mask type {mask_type},"
                "but no conversion strategies exist for this type."
                "Valid mask types are ['categorical', 'boolean']"
            )
    else:
        raise ValueError(
            f"No mask type was found. Ensure {mask.name} has a valid 'mask_type' attribute."
        )

    return masks_ds_bool
