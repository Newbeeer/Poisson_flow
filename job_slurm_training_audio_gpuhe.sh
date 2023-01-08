#!/bin/bash

#SBATCH -n 32
#SBACTH --mem-per-cpu=4G
#SBATCH --gpus=4
#SBATCH --gres=gpumem:20g
#SBATCH --nodes=1
#SBATCH --time=100:00:00
wandb online
python3 main.py --conf sd_128 --mode train --workdir sd_128 --DDP
