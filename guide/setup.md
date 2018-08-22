# Software setup

> #### Questions
>
> 1.   What do I need to run the software?
>
> #### Objectives
>
> 1.   Provide instructions for installing the software.

### Download the ML4Bio software
To download the ML4Bio software, visit <https://github.com/gitter-lab/ml-bio-workshop/>.
Click the `Clone or download` button followed by `Download ZIP`.

<p align="center">
<img width="350" src="../figures/download_button.png">
</p>

Save the file `ml-bio-workshop-master.zip` and then open that location on your computer.
Extract the zip file and open the folder `ml-bio-workshop-master` with the same contents as <https://github.com/gitter-lab/ml-bio-workshop/>.
You are now ready to install the Python dependencies needed to run ML4Bio.

### Install Python
ML4Bio requires Python and several Python packages.
The easiest way to install Python and the correct version of these packages is through [Anaconda](https://anaconda.com/), a Python distribution.
If you do not have Anaconda installed, please visit <https://www.anaconda.com/download/> to download and install the Python 3.6 version.
The default installation will add Anaconda to your computer's `PATH` environment variable so that it is accessible from the command line.

### Launch the software

After you install Anaconda, return to the `ml-bio-workshop-master` directory and open the `src` subdirectory.
There are wrapper scripts that will run ML4Bio inside a conda environment, which is a collection of specific versions of Python packages.
If the environment does not already exist, it will be created.
This can take several 5-10 minutes.
For Windows, launch the `ml4bio_conda.bat` script.
For Mac OS or Linux, launch the `ml4bio_conda.sh` script.

Advanced users who already have Python installed can install the [required packages](../src/readme.md) through pip.
Then launch ML4Bio from the command line by executing `python ml4bio.py` from the `src` directory.
