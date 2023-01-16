import os
import json
from tqdm import tqdm

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

from configs.get_configs import get_config
from evaluation.metrics import compute_metrics
from evaluation import evaluate

from utils.classes import DotDict

print("Loading configuration ...")
args = DotDict()
args.workdir = "/cluster/scratch/tshpakov/results/64_deep"
args.checkpoint_dir = "checkpoints/pfgm/64"
args.conf = "64_deep"
args.test = False
args.DDP = False
args.sampling = True
args.config = get_config(args)

args.config.eval.batch_size = 64
args.config.eval.num_samples = 1200
args.config.eval.input_mel = "64"
args.config.eval.save_audio = True
args.config.eval.enable_benchmarking = True

# Sampling params
args.config.sampling.ode_solver = 'rk45' # 'torchdiffeq', 'improved_euler', 'forward_euler'
args.config.sampling.ckpt_number = 200000
args.config.sampling.N = 100
args.config.sampling.z_max = 45
args.config.sampling.z_min = 1e-3
args.config.sampling.upper_norm = 1800
args.config.seed = 500

# 64
#args.config.sampling.N = 100
#args.config.sampling.z_max = 22
#args.config.sampling.z_min = 1e-3
#args.config.sampling.upper_norm = 1800

# Diffwave 
#args.config.sampling.N = 100
#args.config.sampling.z_max = 550
#args.config.sampling.z_min = 1e-3
#args.config.sampling.upper_norm = 1000

args.eval_folder = f"os_{args.config.sampling.ode_solver}_N_{args.config.sampling.N}_zmax_{args.config.sampling.z_max}_zmin_{args.config.sampling.z_min}_un_{args.config.sampling.upper_norm}_seed_{args.config.seed}"

print("Generate samples ... ")
#evaluate.run(args)

metrics = compute_metrics(f"{args.workdir}/ckpt_{args.config.sampling.ckpt_number}/{args.eval_folder}/audio", gt_metrics=True)

# Log metrics
with open(f'{args.workdir}/ckpt_{args.config.sampling.ckpt_number}/{args.eval_folder}/metrics.txt', 'w+') as metric_file:
     metric_file.write(json.dumps(metrics))
