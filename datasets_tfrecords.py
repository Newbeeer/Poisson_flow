import tensorflow as tf
import tensorflow_datasets as tfds

def preprocess_fn_64(d):
    # apply known data schema to decode the bytestrings
    image_size = 16_000 // 256 + 2
    sample = tf.io.parse_single_example(
    d,
    features={
    'mel': tf.io.FixedLenFeature([64*image_size], tf.float32)
    })
    # reshape the flattened list back to tensor
    data = tf.reshape(sample['mel'], (1,64, image_size))
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
    ds = ds.repeat(count=1)
    ds = ds.shuffle(10000)
    ds = ds.map(preprocess_fn_64, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size=256, drop_remainder=True)
    return ds.prefetch(tf.data.experimental.AUTOTUNE)

def get_dataset(file):

    dataset_builder = tf.data.TFRecordDataset(file)
    train_split_name = eval_split_name = 'train'
    # handle tensorflow datasets
    
    train_ds = create_dataset(dataset_builder, train_split_name)
    eval_ds = create_dataset(dataset_builder, eval_split_name)
    return train_ds, eval_ds