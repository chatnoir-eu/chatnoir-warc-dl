from time import sleep

import imageio as iio
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model(
    "models/chollet.h5")  # https://github.com/samon11/meme-classifier/blob/master/chollet.h5

size = (150, 150)

def gen(ending):
    for i in range(5):
        filename = f"data/test{i}.{ending}"
        image = iio.imread(filename, as_gray=False, pilmode="RGB")
        # print("            " if ending=="jpg" else "",filename)
        sleep(1 if bool(["jpg", "png"].index(ending)) != (i <= 2) else 1.7)  # simulate IO slowness
        resized = tf.image.resize(image / 255., size, antialias=True)
        yield resized, filename


def ds(ending):
    return tf.data.Dataset.from_generator(lambda: gen(ending), output_signature=(
    tf.TensorSpec(shape=size + (3,), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.string)))


individual_datasets = [ds(ending) for ending in ["jpg", "png"]]


def get_individual_dataset(i):
    return individual_datasets[i]


range_dataset = tf.data.Dataset.range(len(individual_datasets))
dtype = tf.data.DatasetSpec.from_value(individual_datasets[0])
dataset = range_dataset.interleave(lambda i: tf.py_function(func=get_individual_dataset, inp=[i], Tout=dtype),
                                   cycle_length=len(individual_datasets), deterministic=False,
                                   num_parallel_calls=tf.data.AUTOTUNE)

dataset = dataset.prefetch(tf.data.AUTOTUNE)
dataset = dataset.batch(3)  # todo make batchsize configurable

def prediction(image, url):
    return model(image, training=False), image, url


results_dataset = dataset.map(prediction)  # todo optimize performance
# todo can this strange mapping instead of prediction technique be used with distribution strategies? is it fast?


# todo filter results

# todo output original-sized image

for element in results_dataset.as_numpy_iterator():
    print(element)
# todo save to file(s)
