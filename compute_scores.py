import time
import os
import numpy as np

from tqdm import *

def fid(train_embeddings, sample_embeddings):
    mean_gt = np.mean(train_embeddings, axis=0)
    mean_gen = np.mean(sample_embeddings, axis=0)

    cov_gt = np.cov(train_embeddings.T)
    cov_gen = np.cov(sample_embeddings.T)

    dist = np.linalg.norm(mean_gt - mean_gen) ** 2
    trace = np.trace(cov_gt + cov_gen - 2 * (cov_gt ** 0.5 * cov_gen * cov_gt ** 0.5) ** 0.5) 
    return dist + trace


def compute_metrics(train_embeddings, sample_embeddings):
    metrics = {}

    ### FID ###
    metrics["fid"] = fid(train_embeddings, sample_embeddings)

    return metrics