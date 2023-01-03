import numpy as np
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
from keras.datasets import cifar10
import os
import json
import tensorflow as tf
 
# scale an array of images to a new size
def scale_images(images, new_shape):
 images_list = list()
 for image in images:
    # resize with nearest neighbor interpolation
    new_image = resize(image, new_shape, 0)
    # store
    images_list.append(new_image)
    return np.asarray(images_list)
 
# calculate frechet inception distance
def calculate_fid(model, images1, images2):
 # calculate activation
 act1 = model.predict(images1)
 act2 = model.predict(images2)
 # calculate mean and covariance statistics
 mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
 mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
 # calculate sum squared difference between means
 ssdiff = np.sum((mu1 - mu2)**2.0)
 # calculate sqrt of product between cov
 covmean = sqrtm(sigma1.dot(sigma2))
 # check and correct imaginary numbers from sqrt
 if np.iscomplexobj(covmean):
    covmean = covmean.real
 # calculate score
 fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
 return fid

def load_mels(sample, gt):
    mels = []
    for mel in np.load(sample)['samples']:
        mels.append(mel)

    mels_gt = []
    for mel in np.load(gt)['samples']:
        mels_gt.append(mel)
    
    return mels, mels_gt

def get_fid(mels, mels_gt, input_shape=(75,75,3)):
    # prepare the inception v3 model
    model = InceptionV3(include_top=False, pooling='avg', input_shape=input_shape)
    # load cifar10 images

    # print('Loaded', mels.shape, mels_gt.shape)
    # convert integer to floating point values
    # mels = mels.astype('float32')
    # mels_gt = mels_gt.astype('float32')
    # resize images
    # mels = scale_images(mels, (64,64,1))
    # mels_gt = scale_images(mels_gt,  (64,64,1))
    # print('Scaled', images1.shape, images2.shape)
    # pre-process images

    mels = preprocess_input(mels)
    mels_gt = preprocess_input(mels_gt)

    # calculate fid
    fid = calculate_fid(model, mels, mels_gt)
    print('FID: %.3f' % fid)

    return fid

def zero_pad_for_inception(samples, gt, end_shape=(75,75,3)):
    pad_top = (end_shape[0] - samples[0].shape[0]) // 2
    pad_bot = pad_top
    if (end_shape[0] - samples[0].shape[0])%2:
        pad_bot+=1

    pad_left = (end_shape[1] - samples[0].shape[1]) // 2
    pad_right = pad_left
    if (end_shape[0] - samples[0].shape[0])%2:
        pad_right+=1

    try:
        assert pad_top + pad_bot + samples[0].shape[0] == end_shape[0]
        assert pad_left + pad_right + samples[0].shape[1] == end_shape[1]
    except AssertionError as e:
        print("invalid dimension {} for desired padding {}".format(((pad_top, pad_bot), (pad_left, pad_right)), end_shape))
        raise e

    dim_pad = np.empty((end_shape[0],end_shape[0],end_shape[2]-samples[0].shape[2]))
    samples_pad=[]
    for sample in samples:
        sample_tmp = np.pad(sample[:,:,0], ((pad_top, pad_bot), (pad_left, pad_right)), 'constant', constant_values=(0,0))
        sample_tmp = np.expand_dims(sample_tmp, axis=2)
        samples_pad.append(np.concatenate((sample_tmp, dim_pad), axis=2))
    
    gt_pad=[]
    for gt_mel in gt:
        gt_tmp = np.pad(gt_mel[:,:,0], ((pad_top, pad_bot), (pad_left, pad_right)), 'constant', constant_values=(0,0))
        gt_tmp = np.expand_dims(gt_tmp, axis=2)
        gt_pad.append(np.concatenate((gt_tmp, dim_pad), axis=2))
        
    return samples_pad, gt_pad

def get_stats(sample_file, gt_file):
    samples, gt = load_mels(sample_file, gt_file)
    samples_pad, gt_pad = zero_pad_for_inception(samples, gt)
    samples_tf = tf.convert_to_tensor(samples_pad, dtype='float32')
    gt_tf = tf.convert_to_tensor(gt_pad, dtype='float32')
    stats = {}
    #### FID ####
    stats["FID"] = get_fid(samples_tf, gt_tf)

    return stats
    
def main():
    results_dir = "./results"
    data_dir = "./mels"
    gt_dir = "./mels/gt"

    print("Evaluating samples in {}".format(data_dir))

    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    samples_path = os.path.join(data_dir, "samples")
    gt_file = os.path.join(data_dir, "gt/samples_0.npz")
    for sample in os.listdir(samples_path):
        print("Evaluating sample {}".format(sample))
        sample_path = os.path.join(samples_path, sample)
        stats = get_stats(sample_path, gt_file)

        results_file = os.path.join(results_dir, "{}_results".format(sample.split('.')[0]))
        with open(results_file, "w") as f:
            json.dump(stats , f) 

if __name__=='__main__':
    main()