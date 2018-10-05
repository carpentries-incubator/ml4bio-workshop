## Python environment

Requires:
- Python 3.5
- pandas
- numpy
- sklearn
- matplotlib
- pyqt 5
- scipy
- git (for installing ml4bio from GitHub)

See `conda_env.yml` for one set of compatible package versions.
Create the `ml4bio` [conda environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) with the command `conda env create -f conda_env.yml`.
On Linux, activate the environment with `source activate ml4bio`.
On Windows, activate the environment with `activate ml4bio`.

The full Anaconda installation also provides all required Python dependencies.

## Running

If Anaconda is already installed and on the system path, the ml4bio package can be installed and launched by double-clicking the appropriate script.
If the `ml4bio` environment does not already exist, the script creates it and installs the required Python packages, including ml4bio.
There are different scripts for each operating system:
- `install_launch_linux.sh` is a bash shell script for Linux.
- `install_launch_mac.command` is a bash shell script for Mac OS, but it has not been tested.
- `install_launch_windows.bat` is a batch file for Windows.

If the Python environment has already been configured externally, the scripts are not required.
Type `ml4bio` from the command line to launch the GUI.

## Third party materials
The icons in the `icons` directory were downloaded from http://thenounproject.com under the Creative Commons license.
Instructions on how to give credit to the creators: [link](https://thenounproject.zendesk.com/hc/en-us/articles/200509928-How-do-I-give-creators-credit-in-my-work-)

Add this attribution where appropriate:
Created by sachin modgekar from Noun Project.
