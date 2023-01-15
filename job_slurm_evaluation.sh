#!/bin/bash
#SBATCH -n 8
#SBATCH --gpus=4
#SBATCH --gres=gpumem:12g
#SBATCH --time=3:00:00
python eval.py