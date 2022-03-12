import os

import tensorflow as tf


def iterate_over_sharded_dataset(dataset_dir):
    """
    Iterates over a previously saved dataset.
    """
    dataset = tf.data.experimental.load(dataset_dir)
    try:
        for data in dataset.as_numpy_iterator():
            yield data
    except tf.errors.DataLossError:
        print("End of data shards.")


if __name__ == "__main__":
    dataset_dir = "data/image_raw_export/out/"
    if not os.path.isdir(dataset_dir):
        raise Exception("Please run raw_export.py first to generate the dataset shard files.")

    for each in iterate_over_sharded_dataset(dataset_dir):
        print(each[-1])
