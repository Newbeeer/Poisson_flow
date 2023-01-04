#!/bin/bash
#SBATCH -n 8
#SBACTH --mem-per-cpu=2048
#SBATCH --gpus=1
#SBATCH --gres=gpumem:16g
#SBATCH --time=00:03:00
wandb offline
python3 main.py --conf sd --mode train --workdir pfgm_sd --test
