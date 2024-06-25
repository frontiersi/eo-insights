# eo-insights

## Conda Environment with Pip
This package uses the approach of creating a conda environment, then pip installing the module.
Dependencies are listed in the `pyproject.toml` file.

### Clone the repository
In your terminal, run 
```
git clone https://github.com/frontiersi/eo-insights.git
```

### Set up
```
conda create -n "eoinsights" python=3.11.0
```
### Activate
```
conda activate eoinsights
```

### Install the package

In your terminal, navigate to the cloned repository folder.
Required dependencies are specified in the `pyproject.toml` file, 
and will be installed with pip.

#### Users
For users, we recommend installing the package along with additional dependencies to enhance working in notebooks. 
To do this, run
```
pip install .[notebooks]
```

Alternatively, to install just the required dependencies, run
```
pip install .
```

#### Developers
After activating the conda environment, run
```
pip install -e .
```
to install the package in editable mode.

### Deactivate
When you have finished working, deactivate the environment.

```
conda deactivate
```