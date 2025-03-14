## Python environment

Requires:
- Python 3.12
- pandas
- numpy
- sklearn
- matplotlib
- pyqt 5
- scipy
- [ml4bio](https://pypi.org/project/ml4bio/)

See `conda_env.yml` for one set of compatible package versions.
Create the `ml4bio` [conda environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) with the command `conda env create -f conda_env.yml`.
On Linux, activate the environment with `conda activate ml4bio`.
On Windows, activate the environment with `activate ml4bio`.
After the environment is created, type `ml4bio` from the command line to launch the software.

## Running
The scripts below may not be compatible with newer versions of Anaconda.

If Anaconda is already installed and on the system path, the ml4bio package can be installed and launched by double-clicking the appropriate script.
If the `ml4bio` environment does not already exist, the script creates it and installs the required Python packages, including ml4bio.
This requires internet connectivity to download the packages.
There are different scripts for each operating system:
- `install_launch_linux.sh` is a bash shell script for Linux.
- `install_launch_mac.command` is a bash shell script for Mac OS.
- `install_launch_windows.bat` is a batch file for Windows.

You may need to make the script executable.
If you have trouble launching the script, try running it in the terminal for Linux or Mac OS or in the Anaconda Prompt (formerly Anaconda Command Prompt) for Windows.

If the Python environment has already been configured externally, the scripts are not required.
Type `ml4bio` from the command line to launch the GUI.

## Warnings

The following warning appears for some combinations of the required Python packages.
It does not affect the ml4bio software:
```
DeprecationWarning: numpy.core.umath_tests is an internal NumPy module and should not be imported. It will be removed in a future NumPy release.
  from numpy.core.umath_tests import inner1d
```
