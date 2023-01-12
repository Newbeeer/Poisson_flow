#!/bin/bash

#SBATCH -n 18
#SBACTH --mem-per-cpu=4GB
#SBATCH --gpus=rtx_2080_ti:8
#SBATCH --time=100:00:00
wandb online
python3 main.py --conf diffwave --mode train --workdir diffwave --DDP
