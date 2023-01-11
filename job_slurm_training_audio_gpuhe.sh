#!/bin/bash

#SBATCH -n 16
#SBACTH --mem-per-cpu=8G
#SBATCH --gpus=2
#SBATCH --gres=gpumem:20g
#SBATCH --nodes=1
#SBATCH --time=100:00:00
wandb online
python3 main.py --conf 128_deep --mode train --workdir pfgm_128_deep_v2 --DDP
