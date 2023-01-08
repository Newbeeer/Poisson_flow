#!/bin/bash

#SBATCH -n 8
#SBACTH --mem-per-cpu=4G
#SBATCH --gpus=4
#SBATCH --gres=gpumem:10g
#SBATCH --nodes=1
#SBATCH --time=100:00:00
wandb online
python3 main.py --conf 128_deep --mode train --workdir pfgm_128_deep_v2 --DDP
