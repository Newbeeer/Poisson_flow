#!/bin/bash

#SBATCH -n 16
#SBATCH --gpus=rtx_2080_ti:8
#SBATCH --time=72:00:00
wandb online
python3 main.py --config ./configs/poisson/audio_ddpmpp_128_deep.py --mode train --workdir pfgm_sc09_128_deep
