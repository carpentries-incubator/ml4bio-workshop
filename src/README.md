## Python environment

Requires:
- Python 3.5
- pandas
- numpy
- sklearn
- matplotlib
- pyqt 5
- scipy

See `conda_env.yml` for one set of compatible package versions.
Create the `ml4bio` [conda environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) with the command `conda env create -f conda_env.yml`.
On Linux, activate the environment with `source activate ml4bio`.

The full Anaconda installation also provides all required dependencies.

## Running

`python3 ml4bio.py` or if you installed the `ml4bio` conda environment use `python ml4bio.py`.

`ml4bio_conda.bat` is a Windows batch file.
If Anaconda is already installed, the GUI can be launched by double-clicking the batch file.
If the `ml4bio` environment does not exist, the batch file creates it and installs the required Python packages.

`ml4bio_conda.sh` is a bash shell script for Linux and Mac OS, but it has not been tested.

## Third party materials
The icons in the `icons` directory were downloaded from http://thenounproject.com under the Creative Commons license.
Instructions on how to give credit to the creators: [link](https://thenounproject.zendesk.com/hc/en-us/articles/200509928-How-do-I-give-creators-credit-in-my-work-)

Add this attribution where appropriate:
Created by sachin modgekar from Noun Project.
