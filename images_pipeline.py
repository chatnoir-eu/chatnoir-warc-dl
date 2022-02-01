from time import sleep

import imageio as iio
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model(
    "models/chollet.h5")  # https://github.com/samon11/meme-classifier/blob/master/chollet.h5

size = (150, 150)

input_image = keras.Input(type_spec=tf.TensorSpec(shape=(None,) + size + (3,), dtype=tf.float32))
input_url = keras.Input(type_spec=tf.TensorSpec(shape=(None,), dtype=tf.string))
pred = model(input_image, training=False)

outputs = {'image': input_image,
           'url': input_url,
           'prediction': pred
           }
modified_model = keras.Model(inputs=[input_image, input_url], outputs=outputs)

def gen(ending):
    for i in range(5):
        filename = f"data/test{i}.{ending}"
        image = iio.imread(filename, as_gray=False, pilmode="RGB")
        # print("            " if ending=="jpg" else "",filename)
        sleep(1 if bool(["jpg", "png"].index(ending)) != (i <= 2) else 1.7)  # simulate IO slowness
        yield tf.image.resize(image, size, antialias=True), filename


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

dataset = dataset.map(lambda image, url: ((image, url),))

dataset = dataset.prefetch(tf.data.AUTOTUNE)
dataset = dataset.batch(10)  # todo make batchsize configurable

# todo we need to export - as a callback?

prediction = modified_model.predict(dataset)

print(prediction["url"])
