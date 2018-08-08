#!/bin/bash
# this script assumes that the user has Anaconda installed somewhere on their
# system and it is on the path

# check whether the user already installed the ml4bio conda environment
# by trying to activate the environment
if !(source activate ml4bio); then
  echo creating ml4bio environment
  conda env create -f conda_env.yml
  source activate ml4bio
fi

# launch the GUI
python ml4bio.py
