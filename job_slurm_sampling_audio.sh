#!/bin/bash
#SBATCH -n 8
#SBATCH --gpus=1
#SBATCH --gres=gpumem:10g
wandb offline
python3 main.py --conf 128_deep --mode eval --workdir pdfg_128_deep_v2 --ckpt 470000 --sampling
