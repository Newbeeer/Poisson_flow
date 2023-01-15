import time
import numpy as np

from scipy.special import softmax
from scipy.stats import entropy

import pickle
from sklearn.cluster import KMeans
import scipy.stats as st

from evaluation.resnext.resnext_utils import generate_embeddings, generate_label_distribution

from evaluation.utils.constants import RESNEXT_CHECKPOINT_PATH, RESNEXT_TRAIN_EMBEDDINGS_PATH, RESNEXT_TRAIN_LOGITS_PATH, KMEANS_MODEL_PATH

def kl(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def fid(train_embeddings, sample_embeddings):
    mean_gt = np.mean(train_embeddings, axis=0)
    mean_gen = np.mean(sample_embeddings, axis=0)

    cov_gt = np.cov(train_embeddings.T)
    cov_gen = np.cov(sample_embeddings.T)

    dist = np.linalg.norm(mean_gt - mean_gen) ** 2
    trace = np.trace(cov_gt + cov_gen - 2 * (cov_gt ** 0.5 * cov_gen * cov_gt ** 0.5) ** 0.5) 
    return dist + trace


def inception_score(label_dist):
    softmax_values = softmax(label_dist, axis=1)
    divs = []
    for softmax_val in softmax_values:
        divs.append(kl(softmax_val, np.mean(softmax_values, axis=0)))
    return np.exp(np.mean(divs))


def modified_inception_score(label_dist):
    softmax_values = softmax(label_dist, axis=1)
    mis = []
    for sv1 in softmax_values:
        for sv2 in softmax_values:
            mis.append(kl(sv1, sv2))
    return np.exp(np.mean(mis))


def am_score(train_label_dist, sample_label_dist):
    softmax_values_train = softmax(train_label_dist, axis=1)
    softmax_values_samples = softmax(sample_label_dist, axis=1)
    am = kl(np.mean(softmax_values_train, axis=0), np.mean(softmax_values_samples, axis=0))
    return am + np.mean(entropy(softmax_values_samples, axis=1), axis=0)


def ndb(train_embeddings, sample_embeddings):
    kmeans = None
    with open(KMEANS_MODEL_PATH, "rb") as f:
        kmeans = pickle.load(f)

    gt_bins = kmeans.predict(train_embeddings)
    gen_bins = kmeans.predict(sample_embeddings)

    gt_bins, gt_counts = np.unique(gt_bins, return_counts=True)
    gen_bins, gen_counts = np.unique(gen_bins, return_counts=True)

    for gt_bin in gt_bins:
        if not gt_bin in gen_bins:
            gen_counts = np.insert(gen_counts, gt_bin, 0)

    counts_per_bin = []
    counts_per_bin_p = []
    counts_per_bin_q = []
    total = 0
    total_q = 0
    total_p = 0
    for gt_bin in gt_bins:
        counts_per_bin_p.append(gt_counts[gt_bin])
        counts_per_bin_q.append(gen_counts[gt_bin])
        counts_per_bin.append(gt_counts[gt_bin] + gen_counts[gt_bin])
        
        total += gt_counts[gt_bin] + gen_counts[gt_bin]
        total_p += gt_counts[gt_bin]
        total_q += gen_counts[gt_bin]

    ps = np.array(counts_per_bin) / total
    ps_p = np.array(counts_per_bin_p) / total_p
    ps_q = np.array(counts_per_bin_q) / total_q

    ses = []
    for p, n_p, n_q in zip(ps, counts_per_bin_p, counts_per_bin_q):
        if n_q == 0 or n_p == 0:
            ses.append(float("inf"))
        else:
            ses.append(np.sqrt(p * (1 - p) * (1 / n_q + 1 / n_p)))

    zs = (ps_q - ps_p) / ses

    alpha = 0.05 # TODO: Add as param

    upper = st.norm.ppf(1 - alpha)
    lower = st.norm.ppf(alpha)

    return np.sum((np.array(zs) > upper) | (np.array(zs) < lower) | (np.isinf(ses))) / 10 # TODO: Add as param


def compute_metrics(audio_path):
    # Compute embeddings and label distribution
    embeddings_train = np.load(RESNEXT_TRAIN_EMBEDDINGS_PATH)
    embeddings_samples = generate_embeddings(RESNEXT_CHECKPOINT_PATH, audio_path)

    label_distribution_train = np.load(RESNEXT_TRAIN_LOGITS_PATH)
    label_distribution_samples = generate_label_distribution(RESNEXT_CHECKPOINT_PATH, audio_path)

    # Compute metrics
    metrics = {}

    metrics["fid"] = fid(embeddings_train, embeddings_samples)
    metrics["is"] = inception_score(label_distribution_samples)
    metrics["mis"] = modified_inception_score(label_distribution_samples)
    metrics["am"] = am_score(label_distribution_train, label_distribution_samples)
    metrics["ndb"] = ndb(embeddings_train, embeddings_samples)

    return metrics