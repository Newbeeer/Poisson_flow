#!/bin/bash

#SBATCH -n 8
#SBATCH --gpus=1
#SBATCH --gres=gpumem:16g
#SBATCH --time=04:00:00
python3 main.py --config ./configs/poisson/audio_ddpmpp.py --mode train --workdir pfgm_sc09
