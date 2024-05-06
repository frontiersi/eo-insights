def calculate_indices(data):

    # calculate indicies
    # Modified Soil Adjusted Vegetation Index (MSAVI)
    msavi = (
        2 * data.nir + 1 - ((2 * data.nir + 1) ** 2 - 8 * (data.nir - data.red)) ** 0.5
    ) / 2
    data["msavi"] = msavi
    # Built-up Area Extraction Index (BAEI)
    baei = (data.red + 0.3) / (data.green + data.swir_1)
    data["baei"] = baei
    # Bare Soil Index (BSI)
    bsi = ((data.swir_1 + data.red) - (data.nir + data.blue)) / (
        (data.swir_1 + data.red) + (data.nir + data.blue)
    )
    data["bsi"] = bsi
    # Normalised Difference Water Index (NDWI)
    ndwi = (data.green - data.nir) / (data.green + data.nir)
    data["ndwi"] = ndwi

    return data
