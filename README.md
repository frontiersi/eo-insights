# remote-sensing-tools

## Conda Environment with Pip
This package uses the approach of creating a conda environment, then pip installing the module.
Dependencies are listed in the `pyproject.toml` file.

### Set up
```
cd remote-sensing-tools
conda create -n "remotesensingtools" python=3.11.0
```
### Activate
```
conda activate remotesensingtools
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