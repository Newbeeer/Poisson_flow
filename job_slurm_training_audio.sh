#!/bin/bash

#SBATCH -n 16
#SBATCH --gpus=4
#SBATCH --gres=gpumem:16g
#SBATCH --time=24:00:00
wandb login
python3 main.py --config ./configs/poisson/audio_ddpmpp.py --mode train --workdir pfgm_sc09_v2.2
