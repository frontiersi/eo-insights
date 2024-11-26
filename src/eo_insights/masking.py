"""Functionality for managing data masks"""

import typing
from functools import partial
from random import randint
from typing import Callable, Iterable, Literal, Optional, Tuple, Union

import dask
import numpy as np
import skimage.morphology

# import dask_image.ndmorph  # type: ignore
import xarray
from skimage.morphology import disk

from eo_insights.stac_utils import MaskInfo

XarrayType = Union[xarray.Dataset, xarray.DataArray]
MorphOperation = Literal["dilation", "erosion", "opening", "closing"]
MORPHOPERATIONS = typing.get_args(MorphOperation)

FlagDefValues = dict[str, Union[bool, str]]
FlagDefBits = Union[int, list[int]]
FlagDef = dict[str, Union[FlagDefBits, FlagDefValues]]
FlagsDefinition = dict[str, FlagDef]

MaskFilter = Tuple[MorphOperation, int]


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
            default_masking_settings=mask_info.default_masking_settings,
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


# Binary masking utils
def set_value_at_index(bitmask: int, index: int, value: bool) -> int:
    """
    Set a bit value onto an integer bitmask

    For example, starting with 0,
    >>> mask = 0
    >>> print(bin(mask))
    0b0

    Set bits 2 and 4 to True
    >>> mask = set_value_at_index(mask, 2, True)
    >>> mask = set_value_at_index(mask, 4, True)
    >>> print(bin(mask))
    0b10100

    Then set bit 2 to False
    >>> mask = set_value_at_index(mask, 2, False)
    >>> print(bin(mask))
    0b10000

    Parameters
    ----------
    bitmask : int
        An existing integer bitmask
    index : int
        The index of the binary bitmask that will be set
    value : bool
        The value to set it as (True or False)

    Returns
    -------
    int
        The updated bitmask
    """

    bit_value = 2**index

    if value:
        # For True, perform bitwise OR on the bitmask and the bit value
        bitmask |= bit_value
    else:
        # For False, perform bitwise AND on the bitmask and the inversion of the bit value
        bitmask &= ~bit_value

    return bitmask


def get_flag_information(
    flag: str, flags_definition: FlagsDefinition
) -> Tuple[list[int], dict[str, bool | str]]:
    """
    For a given flag, return the flag bits and values from the flags_definition.
    This should only be used for bitmask types.

    Parameters
    ----------
    flag : str
        Flag to extract bits and values for
    flags_definition : FlagsDefinition
        Dictionary of the structure: {flag: {bits: [], values: {}}}

    Returns
    -------
    Tuple[list[int] | dict[str, bool | str]]
        A tuple of (bits, values), for example
        bits = [6, 7], values = {'0': "climatology aerosol", '1': "low aerosol"}

    Raises
    ------
    KeyError
        flags_definition has no flag: "flag"
    KeyError
        "flag" in flags_definition has no key 'bits'. Check configuration.
    KeyError
        "flag" in flags_definition has no key 'values'. Check configuration."
    """

    bits_and_values: FlagDef | None = flags_definition.get(flag, None)  # type: ignore
    if bits_and_values is None:
        raise KeyError(f"flags_definition has no flag: '{flag}'.")

    # Get bits used by this flag
    bits: FlagDefBits | None = bits_and_values.get("bits", None)  # type: ignore
    if bits is None:
        raise KeyError(
            f"'{flag}' in flags_definition has no key 'bits'. Check configuration."
        )

    # Convert bits from any to list of integers
    if isinstance(bits, int):
        bits_list = [bits]
    else:
        bits_list = bits

    values: FlagDefValues | None = bits_and_values.get("values", None)  # type: ignore
    if values is None:
        raise KeyError(
            f"'{flag}' in flags_definition has no key 'values'. Check configuration."
        )

    return (bits_list, values)


def get_flag_value_for_option(
    requested_option: str | bool, flag_values: FlagDefValues
) -> int:
    """
    Determine the integer that corresponds to the flag option requested by the user

    Parameters
    ----------
    requested_option : str
        Flag requested by the user, e.g. "low aerosol"
    flag_values : FlagDefValues
        A dictionary mapping strings of integers to options
        e.g. {'0': "climatology aerosol", '1': "low aerosol"}


    Returns
    -------
    int
        The value to use for bitmasking

    Raises
    ------
    KeyError
        No appropriate value was found.
    """

    value = next(
        (
            int(value)
            for value, option in flag_values.items()
            if option == requested_option
        ),
        None,
    )

    if value is None:
        raise KeyError("Value was None please try again")

    return value


def make_reference_mask(bits: list[int]) -> int:
    """
    Create a reference mask, which sets all values listed in

    Parameters
    ----------
    bits : list[int]
        Bits to generate a reference mask for. e.g. [6, 7]

    Returns
    -------
    int
        An integer version of the reference mask.
        e.g. [6, 7] produces a reference mask of 0b01100000, output as 96
    """
    reference_mask = 0
    for bit in bits:
        reference_mask = set_value_at_index(reference_mask, bit, True)

    return reference_mask


def make_reference_value(requested_value: int, bits: list[int]) -> int:
    """
    Calculate the integer that corresponds to the requested value from the user

    Parameters
    ----------
    requested_value : int
        An integer representing the option selected by the user.
        e.g. "low aerosol" has a value of 1 at the relevant bits
    bits : list[int]
        The relevant bits for the requested value
        e.g. [6, 7]

    Returns
    -------
    int
        An integer version of the reference mask.
        e.g. as requested value of 1 at bits [6, 7] produces
        a reference value of 0b00100000, output as 32
    """
    reference_value = requested_value << min(bits)

    return reference_value


def create_mask_from_flag(
    pq_band: xarray.DataArray,
    flag: str,
    option: str | bool,
    flags_definition: FlagsDefinition,
) -> xarray.DataArray:
    """
    Convert from a flag with requested value to a boolean raster mask

    Parameters
    ----------
    pq_band : xarray.DataArray
        Pixel quality band
    flag : str
        The flag to mask
    option : str | bool
        The setting of the flag to mask
    flags_definition : FlagsDefinition
        The flags_definition containing the flag, and its bits and values

    Returns
    -------
    xarray.DataArray
        A boolean raster mask with True for the values that match the option, and False otherwise

    Raises
    ------
    KeyError
        A valid flag option was not provided
    """

    # Extract list of bits, and dict of (value, option) pairs
    # for the chosen flag from the flags definition
    bits, values = get_flag_information(flag, flags_definition)

    # Extract the integer value that maps to the chosen option
    flag_value = get_flag_value_for_option(option, values)

    # Create the binary reference mask
    reference_mask = make_reference_mask(bits)

    # Create the value the mask must equal to produce the chosen option
    reference_value = make_reference_value(flag_value, bits)

    # Perform bitwise and to identify all pixels in pq_band
    # that satisfy the chosen option (represented by the reference value)
    # for the reference mask
    mask = pq_band & reference_mask == reference_value

    return mask


def generate_bit_mask(
    pq_band: xarray.DataArray,
    mask_settings: dict[str, str | bool],
    flags_definition: FlagsDefinition,
):
    """
    For all given mask settings, compute the boolean raster mask where any conditions are true
    Masks are generated for each setting and then combined using bitwise OR

    Parameters
    ----------
    pq_band : xarray.DataArray
        Pixel quality band
    mask_settings : dict[str, str | bool]
        A dictionary of settings for the mask. e.g. {"cloud": True, "aerosol level": "high aerosol"}
    flags_definition : FlagsDefinition
        The flags_definition containing the flag, and its bits and values

    Returns
    -------
    xarray.DataArray
        A boolean raster mask with True where the pixel quality band matches any of the mask settings


    Raises
    ------
    ValueError
        The mask failed to generate.
    """
    mask = None

    for flag, option in mask_settings.items():
        flag_mask = create_mask_from_flag(pq_band, flag, option, flags_definition)
        if mask is None:
            mask = flag_mask
        else:
            mask = mask | flag_mask

    if mask is None:
        raise ValueError(
            "The bit mask was not generated."
            "Check mask settings and flags definition and try again."
        )

    return mask


# def generate_bit_mask(mask: xarray.DataArray, categories: list[str], category_values: dict[str, dict]):


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
            masking_settings = mask.attrs.get("default_masking_settings", {})
            mask_categories = [key for key, _ in masking_settings.items()]
            if len(mask_categories) == 0:
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

            masks_ds_bool = generate_categorical_mask(
                mask=mask,
                categories=mask_categories,
                category_values=mask_category_values,
            )
        elif mask_type == "bitflags":
            # Get settings from metadata
            masking_settings = mask.attrs.get("default_masking_settings")
            if masking_settings is None:
                raise KeyError(
                    f"Mask band {mask.name} has no default masking settings."
                    "Check metadata for default_masking_settings."
                )
            mask_flags_definition = mask.attrs.get("flags_definition")
            if mask_flags_definition is None:
                raise KeyError(
                    f"Mask band {mask.name} has no flag definitions. Check metadata for flags_definitions."
                )

            masks_ds_bool = generate_bit_mask(
                pq_band=mask,
                mask_settings=masking_settings,
                flags_definition=mask_flags_definition,
            )

        elif mask_type == "boolean":
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
