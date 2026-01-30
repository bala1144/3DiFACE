#!/bin/bash
echo "Creating mdm env"
conda env create -n mdm38 --file env38.yml
conda activate mdm38
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
conda install -c conda-forge mpi4py