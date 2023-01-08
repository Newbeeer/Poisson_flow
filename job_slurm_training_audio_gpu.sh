#!/bin/bash

#SBATCH -n 16
#SBACTH --mem-per-cpu=4G
#SBATCH --gpus=rtx_2080_ti:8
#SBATCH --time=100:00:00
wandb online
python3 main.py --conf 128_deep --mode train --workdir pfgm_128_deep_v2 --DDP
