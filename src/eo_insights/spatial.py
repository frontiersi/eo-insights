"""
Spatial tools

Adapted from Digital Earth Australia Tools
"""

from typing import Hashable, Iterable, Optional, Tuple, Union

import geopandas
import numpy
import rasterio
import rasterio.features
import shapely
import shapely.geometry
import xarray
from odc.geo.geobox import GeoBox
from odc.geo.xr import wrap_xr

# Construct types for type hinting
XarrayType = Union[xarray.Dataset, xarray.DataArray]
ShapelyGeometryType = Union[shapely.Point, shapely.Polygon]
RasterizeGeometryValueType = Iterable[Tuple[ShapelyGeometryType, Union[int, float]]]


def xr_vectorize(
    da: xarray.DataArray,
    attribute_name: Optional[Hashable] = None,
    mask: Optional[numpy.ndarray] = None,
) -> geopandas.GeoDataFrame:
    """
    Vectorize values from an xarray

    Parameters
    ----------
    da : xarray.DataArray
        The ``xarray.DataArray`` to vectorize. Must be of type
        int16, int32, uint8, uint16, or float32
    attribute_name : Optional[Hashable]
        The name to assign the values from the vectorized array.
        If None, it will default to `da.name`.
        If `da.name` is None, it will defaults to "attribute"
    mask
        ``numpy.ndarray`` of type ``boolean``. Values of ``False``
        are excluded from vectorization.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the values from `da` for each
        shape identified by ``rasterio.features.shapes``

    Raises
    ------
    ValueError
        If `da.odc.crs` is None.
    """

    # Check if provided data array has a CRS
    if da.odc.crs is None:
        raise ValueError("The data array does not have a coordinate reference system.")

    if da.dtype not in [
        bool,
        numpy.int16,
        numpy.int32,
        numpy.uint8,
        numpy.uint16,
        numpy.float32,
    ]:
        raise TypeError(
            f"The data type of the data array is {da.dtype}. "
            "The type must be one of bool, int16, int32, uint8, uint16, or float32."
        )

    # If the array is a boolean, convert it to an unsigned integer
    # as rasterio.features.shapes does not accept booleans
    if da.dtype == bool:
        da = da.astype(numpy.uint8)

    # Vectorize the data array
    vectors = rasterio.features.shapes(
        source=da.data,
        mask=mask,
        transform=da.odc.geobox.transform,
    )

    # Convert the generator into a list
    vectors = list(vectors)

    # Extract the polygon coordinates and values from the list
    polygons = [polygon for polygon, value in vectors]
    values = [value for polygon, value in vectors]

    # Convert polygon coordinates into polygon shapes
    polygons = [shapely.geometry.shape(polygon) for polygon in polygons]

    if (attribute_name is None) and (da.name is not None):
        attribute_name = str(da.name)
    else:
        attribute_name = "attribute"

    gdf = geopandas.GeoDataFrame(
        data={attribute_name: values}, geometry=polygons, crs=da.odc.crs
    )

    return gdf


def _rasterio_rasterize(
    shapes: Union[RasterizeGeometryValueType, Iterable[ShapelyGeometryType]],
    geobox: GeoBox,
    **rasterio_kwargs,
) -> numpy.ndarray:
    raster = rasterio.features.rasterize(
        shapes=shapes, out_shape=geobox.shape, transform=geobox.transform
    )

    return raster


def xr_rasterize(
    gdf: geopandas.GeoDataFrame,
    da: XarrayType,
    attribute_name: Optional[Hashable] = None,
) -> xarray.DataArray:
    # Check if provided data array has a CRS
    if da.odc.crs is None:
        raise ValueError("data array does not have a coordinate reference system.")

    gdf_reprojected = gdf.to_crs(crs=da.odc.crs)

    if attribute_name is not None:
        shapes = zip(gdf_reprojected.geometry, gdf_reprojected[attribute_name])
    else:
        shapes = gdf_reprojected.geometry

    # Rasterize into a numpy array
    raster = _rasterio_rasterize(shapes=shapes, geobox=da.odc.geobox)

    # Convert to xarray
    da_rasterized = wrap_xr(im=raster, gbox=da.odc.geobox)
    if attribute_name is not None:
        da_rasterized = da_rasterized.rename(attribute_name)

    return da_rasterized
