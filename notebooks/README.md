# Notebooks

This folder contains various notebooks that demonstrate the functionality of the eo-insights package. 

## Requirements

To run these notebooks, you must install the package with the optional `notebooks` dependencies using

```
pip install .[notebooks]
```

## Recommended viewing order
We recommend browsing the notebooks in the following order.

### General functionality for loading data from STAC catalogs

1. [load_demo.ipynb](load_demo.ipynb)
1. [stac_catalog_demo.ipynb](stac_catalog_demo)
1. [masking_demo.ipynb](masking_demo.ipynb)

### Additional functionality after loading data

1. [band_indices.ipynb](band_indices_demo.ipynb)
1. [spatial_tools.ipynb](spatial_tools_demo.ipynb)

### Advanced topics

1. [configuration_demo.ipynb](configuration_demo.ipynb)

### Machine learning
To demonstrate how this package can be used to support EO-based analytics, we provide two machine learning notebooks that were inspired by work undertaken at FrontierSI. 
These notebooks are for demonstration purposes only, and use small subsets of free and open data from Australian databases.
The models developed in these notebooks are not fit for use outside of this demonstration.

1. [machine_learning_habitat_mapping_demo.ipynb](machine_learning_habitat_mapping_demo.ipynb)
1. [machine_learning_species_mapping_demo.ipynb](machine_learning_habitat_mapping_demo.ipynb)