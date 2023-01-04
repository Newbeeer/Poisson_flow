#!/bin/bash

#SBATCH -n 16
#SBATCH --gpus=2
#SBATCH --gres=gpumem:20g
#SBATCH --time=24:00:00
wandb online
python3 main.py --config ./configs/poisson/audio_sd.py --mode train --workdir pfgm_sc09_sd
