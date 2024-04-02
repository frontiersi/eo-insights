# python-datascience-setup

## What this repo/template does
- This repo provides a basic starting point for python-based data science projects. 
  - It uses Conda to help manage our dependencies which often rely on C++ projects like GDAL, and so conda does a great job of managing those dependencies
  - It provides some linting services with VS Code to ensure code is kept nice and tidy. These will automatically be applied to your VS Code workspace (project), but will not change your VS Code user (global) settings

**Note:** This repo is just some *suggested* standards, if you do things differently that's ok 😃

## Starting a new project using this template
- Click the 'Use this template' button on the [repo homepage](https://github.com/frontiersi/python-datascience-setup) and follow the subsequent pages to create your own repo based on this template.
  - **Note:** Any changes you make to your repository won't effect the template repository
- Clone your newly created repository to your local machine.
- You'll probably want to start by modifying the environment name. Open the environment.yml locally and edit the `name` field to something that reflects your project.
- If you have any packages that you specifically want to install for your project, add them under the `# Project-Specific Packages` comment in the environment.yml file
- Create the conda environment for your project  
````
cd your-new-repo-directory
conda env create -f environment.yml
````
- You should now be up and running with a new repo and conda environment for your project 🎉


## Updating this repo/template
- Members of the "FrontierSI staff" Github team have permissions to update this repository. 
- Any changes made to this repo won't effect projects that have been initiated from this template.
