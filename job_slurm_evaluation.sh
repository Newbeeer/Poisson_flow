#!/bin/bash
#SBATCH -n 6
#SBATCH --gpus=1
#SBATCH --gres=gpumem:12g
#SBATCH --time=3:00:00
python eval.py