---
title: Setup
---
# Software overview

### Setup and launch the ml4bio software

> #### Questions
>
> 1.   What do I need to run the ml4bio software?
> 2.   How do I download the ML4Bio workshop materials?
>
> #### Objectives
>
> 1.   Provide instructions for installing the ml4bio software.
> 2.   Explain the ml4bio software environment

### Overview
There are three main steps to prepare for the ML4Bio workshop:
1. [Download the workshop materials](#download-the-ml4bio-materials)
2. [Install the Anaconda Python distribution](#install-python)
3. [Install the ml4bio software](#launch-the-ml4bio-software)

See the [troubleshooting](#troubleshooting) if you run into problems during the installation.
If you already have Python installed and do not want to use Anaconda, download the ML4Bio materials and proceed to the [advanced](#advanced-users) instructions.

### Download the ML4Bio materials
To download the ML4Bio materials, visit <https://github.com/carpentries-incubator/ml4bio-workshop/>.
Click the `Code` button followed by `Download ZIP`.

<p align="center">
<img width="350" src="{{ page.root }}/fig/setup/download_button.png">
</p>

Save the file `ml4bio-workshop-gh-pages.zip` and then open that location on your computer.
Extract the zip file and open the folder `ml4bio-workshop-gh-pages`, which has the same contents as <https://github.com/carpentries-incubator/ml4bio-workshop/>.
You are now ready to install the Python dependencies needed to run the ml4bio software and follow the workshop exercises.
You will also use the datasets in the `data` subdirectory during the workshop.
Take note of the location of the `ml4bio-workshop-gh-pages` folder so you can navigate to it during the workshop.

### Install Python
ml4bio requires Python and several other Python packages.
The easiest way to install Python and the correct version of these packages is through [Anaconda](https://anaconda.com/), a Python distribution.
If you do not have Anaconda installed, please visit <https://www.anaconda.com/download/> to download and install the Python 3.x version (for example, 3.8).
**We recommend letting the installer add Anaconda to your computer's `PATH` environment variable so that it is easily accessible from the command line.**
This screenshot shows the `PATH` option in the 2019 version of the Anaconda Windows installer:

<p align="center">
<img width="350" src="{{ page.root }}/fig/setup/anaconda_path.png">
</p>

This will also make Anaconda your primary Python distribution.
See the [Carpentries Anaconda installation instructions](http://carpentries.github.io/workshop-template/#python) for a step-by-step guide and video on how to install Anaconda for your operating system.

### Launch the ml4bio software
After you install Anaconda, you will use installation scripts in the `scripts` subdirectory of the `ml4bio-workshop-gh-pages` directory to install the ml4bio software.
These are wrapper scripts that will run ml4bio inside a [conda environment](#software-environment-details).
If the environment does not already exist, it will be created.
This can take 5-10 minutes and requires internet connectivity to download the Python packages.
- For **Windows**, launch the `install_launch_windows.bat` script.
You may need to run this script twice, once to install the software and again to launch it.
- For **Mac OS**, launch the `install_launch_mac.command` script.
- For **Linux**, launch the `install_launch_linux.sh` script.

To launch the correct script for your operating system, navigate to the `scripts` subdirectory of the unzipped `ml4bio-workshop-gh-pages` directory from the command line.
For Windows, launch the Anaconda Prompt (formerly Anaconda Command Prompt) and then run the script:
- Start -> Type "Anaconda" -> Anaconda Prompt
- Navigate to the `ml4bio-workshop-gh-pages\scripts` directory from the command line using the command `cd <PATH_TO_ml4bio-workshop-gh-pages>\scripts` (replace `<PATH_TO_ml4bio-workshop-gh-pages>` with the appropriate directory on your computer)
- Type `install_launch_windows.bat` -> Enter

<p align="center">
<img width="500" src="{{ page.root }}/fig/setup/windows_script.png">
</p>

For Linux or Mac OS, open the terminal and navigate to the `ml4bio-workshop-gh-pages/scripts` directory.
Then, enter `./` followed by the name of the script for your operating system without a space in between.

<p align="center">
<img width="500" src="{{ page.root }}/fig/setup/linux_script.png">
</p>

If the ml4bio software was successfully installed, you should see this graphical interface:

<p align="center">
<img width="500" src="{{ page.root }}/fig/software/launch.png">
</p>

After you close the ml4bio software, you can run the same `install_launch` script to relaunch it.
The script will not install anything new the second time you run it.
It will use Anaconda to open ml4bio.

A dedicated [lesson][episode-t-cells] will provide an introduction to the ml4bio software during the workshop.
See the [software environment details](#software-environment-details) for more information about how the ml4bio software works.


### Troubleshooting
You must extract the contents of the `ml4bio-workshop-gh-pages.zip` workshop materials file.
Even though you may be able to browse the compressed directory to inspect the files, the software installation will not work until the file is unzipped.

If you did not add Anaconda to your `PATH` during installation and would like to, follow these instructions for Windows 10:
- Start -> Type "Path" -> Edit environment variables for your account
- Path -> Edit -> New -> Browse -> Browse to the location where Anaconda was installed and select the Scripts subdirectory -> OK -> OK

When running the `install_launch_windows.bat` install script, Windows may display a warning that the app is from an unknown publisher and may be unsafe to run.
This warning can be ignored.

See also known [software warnings](./scripts/README.md#warnings) that can be safely ignored.

### Updating ml4bio
New versions of the ml4bio software will be periodically released through [PyPI](https://pypi.org/project/ml4bio/).
The [release notes](https://github.com/gitter-lab/ml4bio/releases) describe the changes in each new version.
To install the latest version of ml4bio, run the appropriate update script for your operating system:
- `update_ml4bio_windows.bat`
- `update_ml4bio_mac.command`
- `update_ml4bio_linux.sh`

Run these scripts in the same manner as the install scripts above.

### Software environment details
Anaconda includes software that enables you to run Python programs as well as additional tools for managing software environments, programming in Python, and integrating code with textual descriptions and results in Jupyter notebooks.
The software environments are managed by conda, one of the tools included with Anaconda.
An environment is a collection of specific versions of Python packages.
These are all stored in a directory that conda manages.
Having multiple environments allows you to use different versions of the same package for different projects.

The ml4bio install scripts create a new conda environment.
This environment, which is named `ml4bio`, contains the latest version of the `ml4bio` Python package as well as suitable versions of other Python packages that it requires.
The `ml4bio` code may be incompatible with older or newer versions of the Python packages it uses.
The environment makes it easy for you use a collection of Python packages that work together.

The most important required Python package that ml4bio uses is called [scikit-learn](http://scikit-learn.org/).
This is a popular general purpose machine learning package.
When you use the ml4bio graphical interface, it calls functions in scikit-learn to train classifiers and make predictions.

### Advanced users
Advanced users who already have Python installed can install the [required packages](./scripts) through pip.
Then launch ml4bio from the command line with the command `ml4bio`.

{% include links.md %}
