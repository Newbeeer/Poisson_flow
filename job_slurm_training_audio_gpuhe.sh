#!/bin/bash

#SBATCH -n 24
#SBACTH --mem-per-cpu=2048
#SBATCH --gpus=4
#SBATCH --gres=gpumem:20g
#SBATCH --time=100:00:00
wandb online
python3 main.py --conf 64_deep --mode train --workdir pfgm_64_deep --DDP
