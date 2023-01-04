#!/bin/bash

#SBATCH -n 16
#SBACTH --mem-per-cpu=2048
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --time=24:00:00
wandb online
python3 main.py --config ./configs/poisson/audio_ddpmpp_128_deep.py --mode train --workdir pfgm_sc09_128_deep
