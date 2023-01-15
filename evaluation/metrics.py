import time
import numpy as np
import pickle

from sklearn.cluster import KMeans

from scipy.special import softmax
from scipy.stats import entropy
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

    print(mean_gt.shape)

    cov_gt = np.cov(train_embeddings.T)
    cov_gen = np.cov(sample_embeddings.T)

    dist = np.linalg.norm(mean_gen - mean_gt) ** 2
    trace = np.trace(cov_gt + cov_gen - 2 * np.emath.sqrt(np.matmul(cov_gt, cov_gen)).real)
    print(trace)
    print(dist)
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


def ndb(train_embeddings, sample_embeddings, n_classes=10, alpha=0.05):
    kmeans = None
    with open(KMEANS_MODEL_PATH, "rb") as f:
        kmeans = pickle.load(f)

    gt_bins = kmeans.predict(train_embeddings)
    gen_bins = kmeans.predict(sample_embeddings)

    # Compute counts per bin
    gt_bins, gt_counts = np.unique(gt_bins, return_counts=True)
    gen_bins, gen_counts = np.unique(gen_bins, return_counts=True)

    for gt_bin in gt_bins:
        if not gt_bin in gen_bins:
            gen_counts = np.insert(gen_counts, gt_bin, 0)

    counts_per_bin = gt_counts + gen_counts

    total = np.sum(counts_per_bin)
    total_p = np.sum(gt_counts)
    total_q = np.sum(gen_counts)

    ps = np.array(counts_per_bin) / total
    ps_p = np.array(gt_counts) / total_p
    ps_q = np.array(gen_counts) / total_q

    # Compute standard error per bin
    ses = np.sqrt(ps * (1 - ps) * (1 / gt_counts + 1 / gen_counts))

    # Compute z-scores per bin
    zs = (ps_q - ps_p) / ses

    # Compute upper and lower threshold based on alpha
    upper = st.norm.ppf(1 - alpha)
    lower = st.norm.ppf(alpha)

    # Statistically different if z-score outside of thresholds or a bin is empty
    return np.sum((np.array(zs) > upper) | (np.array(zs) < lower) | (np.isinf(ses))) / n_classes


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
    metrics["train_is"] = inception_score(label_distribution_train)

    metrics["mis"] = modified_inception_score(label_distribution_samples)
    metrics["train_mis"] = modified_inception_score(label_distribution_train)

    metrics["am"] = am_score(label_distribution_train, label_distribution_samples)
    metrics["ndb"] = ndb(embeddings_train, embeddings_samples)

    return metrics