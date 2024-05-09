def calculate_indices(ds, index):
    """
    Parameters
    ----------
    ds: xarray Dataset
        A two-dimensional or multi-dimensional array containing the
        spectral bands required to calculate the index. These bands are
        used as inputs to calculated the selected index.
    index: str or list of strs
        A string giving the name of the index to calculate or a list of
        strings giving the names of the indices to calculate:

        * ``'MSAVI'`` (Modified Soil Adjusted Vegetation Index)
        * ``'BAEI'`` (Built-up Area Extraction Index)
        * ``'BSI'`` (Bare Soil Index)
        * ``'NDWI'`` (Normalised Difference Water Index)

    Returns
    -------
    ds: xarray Dataset
        The original xarray Dataset inputted into the function, with a
        new variable containing the remote sensing index as a DataArray.
    """
    # Dictionary containing remote sensing index band indices
    index_dict = {
        # Modified Soil Adjusted Vegetation Index (MSAVI)
        "msavi": lambda ds: (
            2 * ds.nir + 1 - ((2 * ds.nir + 1) ** 2 - 8 * (ds.nir - ds.red)) ** 0.5
        )
        / 2,
        # data["msavi"] = msavi
        # Built-up Area Extraction Index (BAEI)
        "baei": lambda ds: (ds.red + 0.3) / (ds.green + ds.swir_1),
        # data["baei"] = baei
        # Bare Soil Index (BSI)
        "bsi": lambda ds: ((ds.swir_1 + ds.red) - (ds.nir + ds.blue))
        / ((ds.swir_1 + ds.red) + (ds.nir + ds.blue)),
        # data["bsi"] = bsi
        # Normalised Difference Water Index (NDWI)
        "ndwi": lambda ds: (ds.green - ds.nir) / (ds.green + ds.nir),
        # data["ndwi"] = ndwi}
    }

    # If index supplied is not a list, convert to list. This allows us to
    # iterate through either multiple or single indices in the loop below
    indices = index if isinstance(index, list) else [index]

    # Calculate for each index in the list of indices supplied (indexes)
    for index in indices:

        # Select an index function from the dictionary
        index_func = index_dict.get(str(index))

        if index is None:
            raise ValueError(f"No remote sensing `index` was provided.")

        mult = 1.0
        index_array = index_func(ds / mult)

        output_band_name = index
        ds[output_band_name] = index_array

    return ds
