import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from datasets import get_dataset
from configs.get_configs import get_config
import torchaudio

import run_lib

class DotDict(dict):
    def __getattr__(self, key):
        return self[key]
    def __setattr__(self, key, val):
        if key in self.__dict__:
            self.__dict__[key] = val
        else:
            self[key] = val

print("Loading configuration ... ")
args = DotDict()
args.conf = "128_deep"
args.test = False
args.config = get_config(args)
args.workdir = "evaluation/128_deep"
args.eval_folder = "sampling_params" # Dynamically replace in loop when testing sampling parameters
args.config.eval.batch_size = 1
args.DDP = False
args.config.eval.num_samples = 1000
#args.gpu = 
args.config.sampling.ckpt_number = 270000

print("Running evaluation ... ")
run_lib.evaluate(args)

