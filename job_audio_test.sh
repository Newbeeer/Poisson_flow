#!/bin/bash

#SBATCH -n 16
#SBATCH --gpus=1
#SBATCH --gres=gpumem:10g
#SBATCH --time=00:03:00
wandb offline
python3 main.py --config ./configs/poisson/audio_test.py --mode train --workdir pfgm_test
