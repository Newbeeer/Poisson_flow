# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
import tensorflow as tf
import tensorflow_datasets as tfds
from datasets_torch import get_loader as get_torch_loader


def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


def crop_resize(image, resolution):
  """Crop and resize an image to the given resolution."""
  crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
  h, w = tf.shape(image)[0], tf.shape(image)[1]
  image = image[(h - crop) // 2:(h + crop) // 2,
          (w - crop) // 2:(w + crop) // 2]
  image = tf.image.resize(
    image,
    size=(resolution, resolution),
    antialias=True,
    method=tf.image.ResizeMethod.BICUBIC)
  return tf.cast(image, tf.uint8)


def resize_small(image, resolution):
  """Shrink an image to the given resolution."""
  h, w = image.shape[0], image.shape[1]
  ratio = resolution / min(h, w)
  h = tf.round(h * ratio, tf.int32)
  w = tf.round(w * ratio, tf.int32)
  return tf.image.resize(image, [h, w], antialias=True)


def central_crop(image, size):
  """Crop the center of an image to the given size."""
  top = (image.shape[0] - size) // 2
  left = (image.shape[1] - size) // 2
  return tf.image.crop_to_bounding_box(image, top, left, size, size)


def get_dataset(config, evaluation=False):
  """Create data loaders for training and evaluation.

  Args:
    config: A ml_collection.ConfigDict parsed from config files.
    evaluation: If `True`, fix number of epochs to 1.

  Returns:
    train_ds, eval_ds, dataset_builder.
  """
  # Compute batch size for this worker.
  batch_size = config.training.batch_size if not evaluation else config.eval.batch_size

  # Reduce this when image resolution is too large and data pointer is stored
  shuffle_buffer_size = 10000
  prefetch_size = tf.data.experimental.AUTOTUNE
  num_epochs = None if not evaluation else 1

  # Create dataset builders for each dataset.
  if config.data.dataset == "speech_commands":
    if config.data.category in ['audio', 'mel']:
      train_loader = get_torch_loader(dataset="speech", mode="training",   config=config)
      valid_loader = get_torch_loader(dataset="speech", mode="validation", config=config)
    if config.data.category == 'tfmel':
      dataset_builder = tf.data.TFRecordDataset(config.data.tfrecords_path)
      train_split_name = eval_split_name = 'train'
  else:
    raise NotImplementedError(
      f'Dataset {config.data.dataset} not yet supported.')

  # Customize preprocess functions for each dataset.
  # handle tfrecords decode of mel dataset
  if config.data.dataset == 'speech_commands' and config.data.category == 'tfmel':
    def preprocess_fn(d):
      # apply known data schema to decode the bytestrings
      sample = tf.io.parse_single_example(
        d,
        features={'mel': tf.io.FixedLenFeature([config.data.image_height * config.data.image_width], tf.float32)})
      # reshape the flattened list back to tensor
      data = tf.reshape(sample['mel'], (1, config.data.image_height, config.data.image_width))
      return dict(image=data, label=None)

  def create_dataset(dataset_builder, split):
    dataset_options = tf.data.Options()
    dataset_options.experimental_optimization.map_parallelization = True
    dataset_options.experimental_threading.private_threadpool_size = 48
    dataset_options.experimental_threading.max_intra_op_parallelism = 1
    read_config = tfds.ReadConfig(options=dataset_options)
    # prepare and build tf datasets
    if isinstance(dataset_builder, tfds.core.DatasetBuilder):
      import resource
      low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
      resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))
      dataset_builder.download_and_prepare()
      ds = dataset_builder.as_dataset(split=split, shuffle_files=True, read_config=read_config)
    # else use the tf records dataset
    else:
      ds = dataset_builder.with_options(dataset_options)
    # set repetition
    ds = ds.repeat(count=num_epochs)
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(prefetch_size)

  # handle tensorflow datasets
  if not config.data.dataset in ["speech_commands"] or config.data.category == 'tfmel':
    train_ds = create_dataset(dataset_builder, train_split_name)
    eval_ds = create_dataset(dataset_builder, eval_split_name)
  # handle pytorch datasets
  else:
    train_ds = train_loader
    eval_ds = valid_loader
    dataset_builder = None
  
  return train_ds, eval_ds, dataset_builder
