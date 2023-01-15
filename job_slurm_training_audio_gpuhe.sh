#!/bin/bash

#SBATCH -n 20
#SBACTH --mem-per-cpu=4G
#SBATCH --gpus=4
#SBATCH --gres=gpumem:20g
#SBATCH --nodes=1
#SBATCH --time=100:00:00
wandb online
python3 main.py --conf 128_deep_2 --mode train --workdir pfgm_128_deep_v3 --DDP