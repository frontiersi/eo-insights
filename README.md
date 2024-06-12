# eo-data-tools

## Conda Environment with Pip
This package uses the approach of creating a conda environment, then pip installing the module.
Dependencies are listed in the `pyproject.toml` file.

### Set up
```
cd eo-data-tools
conda create -n "eodatatools" python=3.11.0
```
### Activate
```
conda activate eodatatools
```

### Install the package
After activating the conda environment, run
```
pip install -e .
```

Note that this installs all required dependencies listed in the `pyproject.toml` file.

### Deactivate
```
conda deactivate
```