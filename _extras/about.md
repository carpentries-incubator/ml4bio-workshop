---
title: About ml4bio software
---

See the [ml4bio graphical interface lesson][episode-gui] for a guide to the ml4bio software.

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

The most important required Python package that ml4bio uses is called [scikit-learn](https://scikit-learn.org/).
This is a popular general purpose machine learning package.
When you use the ml4bio graphical interface, it calls functions in scikit-learn to train classifiers and make predictions.

### Advanced users
Advanced users who already have Python installed can install the [required packages](https://github.com/gitter-lab/ml-bio-workshop/tree/gh-pages/scripts) through pip.
Then launch ml4bio from the command line with the command `ml4bio`.

{% include links.md %}
