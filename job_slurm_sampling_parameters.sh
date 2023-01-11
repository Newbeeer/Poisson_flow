#!/bin/bash

#SBATCH -n 8
#SBATCH --gpus=1
#SBATCH --gres=gpumem:10g
#SBATCH --time=8:00:00
python3 sampling_parameters.py
