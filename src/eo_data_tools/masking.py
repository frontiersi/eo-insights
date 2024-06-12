"""Functionality for managing data masks"""

from functools import partial
import logging
from random import randint
import typing
from typing import Union, Literal, Iterable, Tuple, Callable, Optional

import dask
import dask_image.ndmorph  # type: ignore
import skimage.morphology
from skimage.morphology import disk
import numpy as np
import xarray
from eo_data_tools.stac_utils import MaskInfo


XarrayType = Union[xarray.Dataset, xarray.DataArray]
MorphOperation = Literal["dilation", "erosion", "opening", "closing"]
MORPHOPERATIONS = typing.get_args(MorphOperation)

MaskFilter = Tuple[MorphOperation, int]

_log = logging.getLogger(__name__)


def set_mask_attributes(
    mask: xarray.DataArray, mask_info: MaskInfo
) -> xarray.DataArray:
    """
    Attach information in MaskInfo to xarray.DataArray attribute

    Parameters
    ----------
    mask : xarray.DataArray
        Data array from RasterBase.masks
    mask_info : MaskInfo
        MaskInfo dataclass containing metadata about the mask

    Returns
    -------
    xarray.DataArray
        Original mask with content of MaskInfo added to the attributes

    Raises
    ------
    ValueError
        `mask.name` does not equal `MaskInfo.alias`
    """
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
    """
    Generate a categorical mask

    Parameters
    ----------
    mask : xarray.DataArray
        Data array from RasterBase.masks
    categories : list[str]
        Categories to mask. Pixels matching the category will return as True.
    category_values : dict[str, int]
        Dictionary matching the categories to their pixel values.

    Returns
    -------
    xarray.DataArray
        Boolean xarray where pixels matching the categories are True, all other pixels are False
    """

    values = [category_values[category] for category in categories]

    mask_bool = mask.isin(values)

    mask_bool.attrs = mask.attrs
    mask_bool.attrs.update(mask_type="boolean")

    return mask_bool


# Adapted from odc-algo
# https://github.com/opendatacube/odc-algo/blob/f67879b1df951f4e1a3e3d52c13b244d1cb516a7/odc/algo/_masking.py#L337
def _disk(radius: int, n_dimensions: int = 2) -> np.ndarray:
    """
    Wrapper for skimage.morphology.disk for use with multiple dimensional arrays

    Parameters
    ----------
    radius : int
        Radius for the disk
    n_dimensions : int, optional
        Number of dimensions in the array the disk will be applied to, by default 2

    Returns
    -------
    np.ndarray
        Disk kernel of radius, with necessary dimensions
    """

    kernel = disk(radius)
    while kernel.ndim < n_dimensions:
        kernel = kernel[np.newaxis]
    return kernel


def _add_random_token_to_string(prefix: str) -> str:
    """
    Append random token to name

    Parameters
    ----------
    prefix : str
        String to append random token to

    Returns
    -------
    str
        Updated string, with random token appended to prefix
    """
    return f"{prefix}-{randint(0, 0xFFFFFFFF):08x}"


def _get_morph_operator(
    operation: MorphOperation,
    dask_enabled: bool,
) -> Callable:
    """
    Select the correct morphological operations library from skimage or dask_image.
    dask_image library currently disabled as results differ from skimage.

    Parameters
    ----------
    operation : MorphOperation
        Morphological operation. Must be one of "dilation", "erosion", "opening", "closing"
    is_dask_collection : bool
        Whether the operation will be applied to a DaskCollection

    Returns
    -------
    Callable
        The appropriate morphological operation function.
        If dask_enabled, the function comes from dask_image.ndmorph.
        If not dask_enabled, the function comes from skimage.morphology.
    """

    # DISABLED UNTIL BEHAVIOUR MATCHES SKIMAGE
    # dask_operators: dict[MorphOperation, Callable] = {
    #     "dilation": dask_image.ndmorph.binary_dilation,
    #     "erosion": dask_image.ndmorph.binary_erosion,
    #     "opening": dask_image.ndmorph.binary_opening,
    #     "closing": dask_image.ndmorph.binary_closing,
    # }

    skimage_operators: dict[MorphOperation, Callable] = {
        "dilation": skimage.morphology.binary_dilation,
        "erosion": skimage.morphology.binary_erosion,
        "opening": skimage.morphology.binary_opening,
        "closing": skimage.morphology.binary_closing,
    }

    if dask_enabled:
        _log.warning(
            "Attempting to use dask_image library, but this option is currently disabled. Defaulting to skimage instead."
        )
        morph_operator = skimage_operators[operation]  # dask_operators[operation]
    else:
        morph_operator = skimage_operators[operation]

    return morph_operator


def _apply_morph_operator_np(
    mask: np.ndarray,
    operation: MorphOperation,
    radius: int,
    dask_enabled: Optional[bool] = None,
    **kw,
) -> np.ndarray:
    """
    Apply a single morphological operator to a numpy ndarray

    Parameters
    ----------
    mask : np.ndarray
        Mask to apply the operation to
    operation : MorphOperation
        Morphological operation. Must be one of "dilation", "erosion", "opening", "closing"
    radius : int
        Radius (number of pixels) of the disk to use in the morphological operation.
    dask_enabled : Optional[bool], optional
        Optional override by the user, by default None

    Returns
    -------
    np.ndarray
        Mask after the operation has been applied.
    """

    # Default to the array's dask collection status unless overruled by the user
    if dask_enabled is None:
        dask_enabled = dask.is_dask_collection(mask)

    morph_operator = _get_morph_operator(operation=operation, dask_enabled=dask_enabled)

    kernel = _disk(radius, mask.ndim)
    filtered_mask = morph_operator(mask, kernel, **kw)

    return filtered_mask


def apply_morph_operator(
    mask_da: xarray.DataArray,
    operation: MorphOperation,
    radius: int,
    **kw,
) -> xarray.DataArray:
    """
    _summary_

    Parameters
    ----------
    mask_da : xarray.DataArray
        Mask to apply the operation to
    operation : MorphOperation
        Morphological operation. Must be one of "dilation", "erosion", "opening", "closing"
    radius : int
        Radius (number of pixels) of the disk to use in the morphological operation.

    Returns
    -------
    xarray.DataArray
        Mask after the operation has been applied.
    """

    mask = mask_da.data

    if operation not in MORPHOPERATIONS:
        raise ValueError(
            f"Requested morphological operation: `{operation}` was not recognised."
            f"Valid operations are: {MORPHOPERATIONS}"
        )

    filtered_mask = _apply_morph_operator_np(
        mask=mask, operation=operation, radius=radius, **kw
    )

    return xarray.DataArray(
        data=filtered_mask,
        coords=mask_da.coords,
        dims=mask_da.dims,
        attrs=mask_da.attrs,
    )


def _apply_morph_operators_np(
    mask: np.ndarray, mask_filters: Iterable[MaskFilter], **kw
) -> np.ndarray:
    """
    Apply a set of morphological operators to a numpy ndarray

    Parameters
    ----------
    mask : np.ndarray
        Mask to apply the operation to
    mask_filters : Iterable[MaskFilter]
        An iterable of tuples of the form (operation, radius),
        where operation must be one of "dilation", "erosion", "opening", "closing"
        and radius is the size of the disk to be used for the operation

    Returns
    -------
    np.ndarray
        Mask after all operations have been applied.
    """

    # dask_enabled is hard-coded to false for this function
    # this is because this function can be used in apply_morph_operators,
    # which has been designed to work with skimage.morphology functions
    for operation, radius in mask_filters:
        if radius > 0:
            mask = _apply_morph_operator_np(
                mask=mask, operation=operation, radius=radius, dask_enabled=False, **kw
            )

    return mask


def _compute_overlap_depth(r: Iterable[int], ndim: int) -> Tuple[int, ...]:
    """
    Utility function for working with dask

    Parameters
    ----------
    r : Iterable[int]
        Radius for disk kernel
    ndim : int
        Number of dimensions

    Returns
    -------
    Tuple[int, ...]
        n-dimensional tuple with the value of the maximum radius in the final two entries
    """
    _r = max(r)
    return (0,) * (ndim - 2) + (_r, _r)


def apply_morph_operators(
    mask_da: xarray.DataArray,
    mask_filters: Iterable[MaskFilter],
    name: Optional[str] = None,
) -> xarray.DataArray:
    """
    _summary_

    Parameters
    ----------
    mask_da : xarray.DataArray
        Mask to apply the operation to
    mask_filters : Iterable[MaskFilter]
        An iterable of tuples of the form (operation, radius),
        where operation must be one of "dilation", "erosion", "opening", "closing"
        and radius is the size of the disk to be used for the operation

    Returns
    -------
    xarray.DataArray
        Mask after all operations have been applied.
    """

    data = mask_da.data

    if dask.is_dask_collection(data):
        radius_list = [radius for _, radius in mask_filters]
        depth = _compute_overlap_depth(r=radius_list, ndim=data.ndim)

        if name is None:
            name = "apply_morph_operators"
            for radius in radius_list:
                name = name + f"_{radius}"

        data = data.map_overlap(
            partial(
                _apply_morph_operators_np,
                mask_filters=mask_filters,
            ),
            depth,
            boundary="none",
            name=_add_random_token_to_string(name),
        )
    else:
        data = _apply_morph_operators_np(mask=data, mask_filters=mask_filters)

    return xarray.DataArray(
        data, attrs=mask_da.attrs, coords=mask_da.coords, dims=mask_da.dims
    )


def convert_mask_to_bool(masks_ds: xarray.Dataset, mask_name: str) -> xarray.DataArray:
    """
    Convert a single mask_name from masks_ds from its base type to a boolean

    Parameters
    ----------
    masks_ds : xarray.Dataset
        xarray.Dataset containing masks
    mask_name : str
        Name of the mask to convert

    Returns
    -------
    xarray.DataArray
        A boolean version of the selected mask

    Raises
    ------
    KeyError
        `mask_name` not a variable of `masks_ds`
    KeyError
        Selected categorical mask has no `categories_to_mask` attribute.
    KeyError
        Selected categorical mask has no `flags_definition` attribute.
    NotImplementedError
        No conversion strategy exists for the mask's type.
    KeyError
        Selected mask has no `mask_type` attribute.
    """

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
                raise KeyError(
                    f"Mask band {mask.name} has no categories to mask."
                    "Check metadata for categories to mask."
                )

            # Get flags definition values from metadata
            mask_flags_definition = mask.attrs.get("flags_definition")
            if mask_flags_definition is not None:
                mask_category_values = mask_flags_definition.get("values")
            else:
                raise KeyError(
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
        raise KeyError(
            f"No mask type was found. Ensure {mask.name} has a valid 'mask_type' attribute."
        )

    return masks_ds_bool
