#!/bin/bash
################################################################################
### This script assumes that conda has been installed and aliased to 'conda'
################################################################################
# Assume we are in an activated environment with python 3.10, e.g. created by
# mamba create -n protevo python=3.10

# Install pytorch
mamba install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install standard data science packages
mamba install numpy matplotlib pandas scipy tqdm seaborn scikit-learn ipykernel lightning

# Install flash attention
# On kraken, I needed to explicitly install cuda-toolkit
mamba install cuda-toolkit=12.1 -c nvidia
# Required pip packages for flash attention
pip install ninja packaging
pip install flash-attn --no-build-isolation

# CherryML
pip install git+https://github.com/songlab-cal/CherryML

# pip packages
pip install einops fair-esm biopython wandb

############################################################################################################################################################
# Make the project itself a python package (to enable more convenient absolute imports)
# 1) Generate a setup.py file for the project with 'project' as name
# Remark: Enable echo to interpret backslash escapes (required for the two newline characters '\n\n' below)
echo -e "from setuptools import setup, find_packages\n\nsetup(name='protevo_bear', version='1.0', packages=find_packages())" > setup.py

# 2) Install the project itself by its name and make it editable (=> use '-e' for editable)
# Remark: This requires the setup.py file created above
pip install -e .

# 3) Install ipykernel for the environment so we can use it in jupyter lab
python -m ipykernel install --user --name protevo_bear --display-name "Python 3.10 (protevo_bear)"
############################################################################################################################################################

echo " "
echo "Installation done"