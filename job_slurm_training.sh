#!/bin/bash

#SBATCH -n 8
#SBATCH --gpus=1
#SBATCH --gres=gpumem:10g

python3 main.py --config ./configs/poisson/cifar10_ddpmpp.py --mode train --workdir poisson_ddpmpp
