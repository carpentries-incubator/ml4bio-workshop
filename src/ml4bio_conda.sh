#!/bin/bash
# this script assumes that the user has Anaconda installed somewhere on their
# system and it is on the path

# check whether the user already installed the ml4bio conda environment
# by trying to activate the environment
source activate ml4bio

if [ $? -ne 0 ]; then
  echo creating ml4bio environment
  conda env create -f conda_env.yml
  source activate ml4bio
fi

# display the activated conda environment and where it is installed
# to help with troubleshooting
conda info --envs

# launch the GUI
python ml4bio.py
