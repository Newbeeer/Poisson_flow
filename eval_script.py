import numpy as np
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize
import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt

from datasets_tfrecords import get_dataset
import pickle
import tqdm
from time import time
 
# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return np.asarray(images_list)

def load_gt_stats(gt_stats_dir):
    mean = np.load(os.path.join(gt_stats_dir, "gt_mean.pickle"), allow_pickle=True)
    cov = np.load(os.path.join(gt_stats_dir, "gt_cov.pickle"), allow_pickle=True)
    return mean, cov

def calculate_gt_stats(stats_dir):
    pkl_files = [file for file in os.listdir(stats_dir) if file.endswith(".pickle")]
    act_list = []
    for file in pkl_files:
        act_np = np.load(os.path.join(stats_dir, file), allow_pickle=True)
        for act in act_np:
            act_list.append(act)

    act = np.reshape(np.asarray(act_list), (np.asarray(act_list).shape[0]*np.asarray(act_list).shape[1], np.asarray(act_list).shape[2]))
    mean = np.asarray(act).mean(axis=0)
    cov = np.cov(act, rowvar=False)
    with open(os.path.join(stats_dir, "gt_mean.pickle"), 'wb') as f:
        pickle.dump(mean, f)
    
    with open(os.path.join(stats_dir, "gt_cov.pickle"), 'wb') as f:
        pickle.dump(cov, f)

    return mean, cov


def load_sample_mels(samples):
    return [sample for sample in np.load(samples)['samples']]

def calculate_stats_gt(model, stats_dir, gt_dataset):
    train_set, _ = get_dataset(gt_dataset)
    activations = []
    cnt=0

    for batch in train_set:
        print("processing batch {}/151, {}%".format(cnt, np.round(cnt/151*100, decimals=2)))
        # plt.imshow(batch['image'][0,0,:,:])
        # plt.savefig(stats_dir+"/test_pre.png")
        breakpoint()
        batch_tmp = tf.transpose(batch['image'], perm=[0, 2, 3, 1])
        batch_pad = zero_pad_for_inception(batch_tmp)
        act = model.predict(tf.convert_to_tensor(batch_pad))
        activations.append(act)
        if not cnt % 50 and cnt != 0:
            with open(os.path.join(stats_dir, "activations_{}.pickle".format(cnt)), 'wb') as f:
                    pickle.dump(activations, f)
            activations = []
        cnt += 1
    
    with open(os.path.join(stats_dir, "activations_{}.pickle".format(cnt)), 'wb') as f:
                    pickle.dump(activations, f)

def get_gt_stats(gt_stats_dir, model, gt_dataset):
    if not os.path.isdir(gt_stats_dir):
        os.mkdir(gt_stats_dir)
    if any("gt_" in file for file in os.listdir(gt_stats_dir)):
        mu_gt, cov_gt = load_gt_stats(gt_stats_dir)
    elif len(os.listdir(gt_stats_dir)):
        mu_gt, cov_gt = calculate_gt_stats(gt_stats_dir)
    else:
        if gt_dataset is None:
            raise FileNotFoundError("No gt stats data found at {}, need dataset file to recompute, {} was given".format(gt_stats_dir, gt_dataset))
        calculate_stats_gt(model, gt_stats_dir, gt_dataset)
        mu_gt, cov_gt = calculate_gt_stats(gt_stats_dir) 
    
    return mu_gt, cov_gt

def preprocess_samples(sample_file):
    samples = load_sample_mels(sample_file)
    samples_pad = zero_pad_for_inception(samples)
    samples_tf = tf.convert_to_tensor(samples_pad, dtype='float32')
    return preprocess_input(samples_tf)

    
def get_fid(sample_file, gt_stats_dir, gt_dataset=None):
    # load data
    model_inputs = preprocess_samples(sample_file)
    
    # load model
    model = InceptionV3(include_top=False, pooling='avg', input_shape=model_inputs.shape[-3:], weights="imagenet")
    mu_gt, sigma_gt = get_gt_stats(gt_stats_dir, model, gt_dataset)
    breakpoint()
    # calculate activation
    act_s = model.predict(model_inputs)
    # calculate mean and covariance statistics
    mu_s, sigma_s = act_s.mean(axis=0), np.cov(act_s, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu_s - mu_gt)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma_s.dot(sigma_gt))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma_s + sigma_gt - 2.0 * covmean)

    print('FID: %.3f' % fid)
    return fid

def zero_pad_single(samples):
    target_h = 75 if samples.shape[0] < 75 else samples.shape[0]
    target_w = 75 if samples.shape[1] < 75 else samples.shape[1]

    target_shape = (target_h, target_w, 3)

    pad_top = (target_shape[0] - samples.shape[0]) // 2 
    pad_bot = pad_top
    if (target_shape[0] - samples.shape[0])%2:
        pad_bot+=1

    pad_left = (target_shape[1] - samples.shape[1]) // 2
    pad_right = pad_left
    if (target_shape[0] - samples.shape[0])%2:
        pad_right+=1

    try:
        assert pad_top + pad_bot + samples.shape[0] == target_shape[0]
        assert pad_left + pad_right + samples.shape[1] == target_shape[1]
    except AssertionError as e:
        print("invalid dimension {} for desired padding {}".format(((pad_top, pad_bot), (pad_left, pad_right)), target_shape))
        raise e

    dim_pad = np.zeros((target_shape[0],target_shape[0],target_shape[2]-samples.shape[2]))
    sample_tmp = np.pad(samples[:,:,0], ((pad_top, pad_bot), (pad_left, pad_right)), 'constant', constant_values=(0,0))
    sample_tmp = np.expand_dims(sample_tmp, axis=2)
    return np.concatenate((sample_tmp, dim_pad), axis=2)

def zero_pad_for_inception(samples):
    target_h = 75 if samples[0].shape[0] < 75 else samples[0].shape[0]
    target_w = 75 if samples[0].shape[1] < 75 else samples[0].shape[1]

    target_shape = (target_h, target_w, 3)

    pad_top = (target_shape[0] - samples[0].shape[0]) // 2
    pad_bot = pad_top
    if (target_shape[0] - samples[0].shape[0])%2:
        pad_bot+=1

    pad_left = (target_shape[1] - samples[0].shape[1]) // 2
    pad_right = pad_left
    if (target_shape[0] - samples[0].shape[0])%2:
        pad_right+=1

    try:
        assert pad_top + pad_bot + samples[0].shape[0] == target_shape[0]
        assert pad_left + pad_right + samples[0].shape[1] == target_shape[1]
    except AssertionError as e:
        print("Invalid dimension {} for padding {}".format(((pad_top, pad_bot), (pad_left, pad_right)), target_shape))
        raise e

    dim_pad = np.zeros((target_shape[0],target_shape[1],target_shape[2]-samples[0].shape[2]))
    
    if target_shape == (75,75,3):
        samples_pad=[]
        for sample in samples:
            sample_tmp = np.pad(sample[:,:,0], ((pad_top, pad_bot), (pad_left, pad_right)), 'constant', constant_values=(0,0))
            sample_tmp = np.expand_dims(sample_tmp, axis=2)
            samples_pad.append(np.concatenate((sample_tmp, dim_pad), axis=2))
    elif target_shape == (128,128,3):
        samples_pad = [np.concatenate((sample, dim_pad), axis=2) for sample in samples]

    return samples_pad

def get_stats(sample_file, gt_stats_dir, gt_dataset=None):
    stats = {}
    #### FID ####
    stats["FID"] = get_fid(sample_file, gt_stats_dir, gt_dataset)

    return stats
    
def main():
    results_dir = "./eval/results" # where to save evaluation scores
    data_dir = "./eval/samples" # directory with sample npz files
    gt_stats_dir="./eval/gt_stats_64" # directory with gt mean and covariance
    gt_file=None # path to tfrecords file

    print("Evaluating samples in {}".format(data_dir))

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    for sample in os.listdir(data_dir):
        print("Evaluating sample {}".format(sample))
        sample_file = os.path.join(data_dir, sample)
        stats = get_stats(sample_file, gt_stats_dir, gt_dataset=gt_file)

        results_file = os.path.join(results_dir, "{}_results".format(sample.split('.')[0]))
        with open(results_file, "w") as f:
            json.dump(stats , f) 

if __name__=='__main__':
    main()