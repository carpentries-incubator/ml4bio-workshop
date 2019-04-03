#!/bin/bash
# this script assumes that the user has Anaconda installed somewhere on their
# system, it is on the path, and the ml4bio conda environment has already
# been installed
source activate ml4bio

# try updating the ml4bio package
pip install ml4bio --upgrade --upgrade-strategy only-if-needed
