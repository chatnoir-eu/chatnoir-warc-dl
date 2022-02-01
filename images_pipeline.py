import base64
from time import sleep

import imageio as iio
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model(
    "models/chollet.h5")  # https://github.com/samon11/meme-classifier/blob/master/chollet.h5

BATCHSIZE = 3

size = (150, 150)


def gen(ending):
    for i in range(5):
        filename = f"data/test{i}.{ending}"
        image = iio.imread(filename, as_gray=False, pilmode="RGB")
        # print("            " if ending=="jpg" else "",filename)
        sleep(1 if bool(["jpg", "png"].index(ending)) != (i <= 2) else 1.7)  # simulate IO slowness
        resized = tf.image.resize(image / 255., size, antialias=True)
        yield tf.RaggedTensor.from_tensor(image, ragged_rank=2), resized, filename


def ds(ending):
    return tf.data.Dataset.from_generator(lambda: gen(ending), output_signature=(
        tf.RaggedTensorSpec(shape=(None, None, 3), dtype=tf.uint8, ragged_rank=2),
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
dataset = dataset.batch(BATCHSIZE)


# strategy = tf.distribute.MirroredStrategy()
# dataset = strategy.experimental_distribute_dataset(dataset) #todo test by using strategy.run() later


def predict(image, resized, url):
    prediction = model(resized, training=False)
    return prediction, image, url


dataset = dataset.map(predict, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
# todo can this strange mapping instead of prediction technique be used with distribution strategies? is it fast? test if GPU makes it faster

dataset = dataset.unbatch()


def filter(prediction, image, url):
    return tf.reshape(prediction > .9, ())


dataset = dataset.filter(filter)


def ragged_to_tensor(prediction, image, url):
    return prediction, image.to_tensor(), url


dataset = dataset.map(ragged_to_tensor, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)


def export(prediction, image, url):
    prediction = prediction[0]
    print(url.decode("utf-8"), prediction)
    iio.imwrite(f"data/out/{base64.b64encode(url).decode('utf-8')}_{prediction:1.4f}.jpg", image)


for data in dataset.as_numpy_iterator():
    export(*data)

# todo offer alternative dataset saving method
