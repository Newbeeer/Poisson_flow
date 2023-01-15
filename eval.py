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

print("Loading configuration ... ")
args = DotDict()
args.workdir = "pfgm_128_deep_v2"
args.checkpoint_dir = "pfgm_128_deep_v2/checkpoints"
args.conf = "128_deep"
args.test = False
args.DDP = False
args.sampling = True
args.config = get_config(args)

args.config.eval.batch_size = 64
args.config.eval.num_samples = 1200
args.config.eval.input_mel = "128"
args.config.eval.save_audio = True
args.config.eval.enable_benchmarking = True

# Sampling params
args.config.sampling.ode_solver = 'rk45' # 'torchdiffeq', 'improved_euler', 'forward_euler'
args.config.sampling.ckpt_number = 500000
args.config.sampling.N = 100
args.config.sampling.z_max = 25
args.config.sampling.z_min = 1e-3
args.config.sampling.upper_norm = 5000
args.config.seed = 49

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
evaluate.run(args)

metrics = compute_metrics(f"{args.workdir}/ckpt_{args.config.sampling.ckpt_number}/{args.eval_folder}/audio")

# Log metrics
with open(f'{args.workdir}/ckpt_{args.config.sampling.ckpt_number}/{args.eval_folder}/metrics.txt', 'w+') as metric_file:
     metric_file.write(json.dumps(metrics))
