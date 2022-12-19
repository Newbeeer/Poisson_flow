#!/bin/bash

#SBATCH -n 8
#SBATCH --gpus=2
#SBATCH --gres=gpumem:16g
#SBATCH --time=01:00:00
python3 main.py --config ./configs/poisson/cifar10_ddpmpp.py --mode train --workdir poisson_ddpmpp
