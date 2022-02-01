import matplotlib.image
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


def gen():
    # todo this is just a dummy
    image = matplotlib.image.imread("data/dummy.jpg")
    for _ in range(100):
        yield tf.image.resize(image, size, antialias=True), "dummy.jpg"


dataset = tf.data.Dataset.from_generator(gen, output_signature=(tf.TensorSpec(shape=size + (3,), dtype=tf.float32),
                                                                tf.TensorSpec(shape=(), dtype=tf.string)))

dataset = dataset.map(lambda image, url: ((image, url),))

# todo use https://www.tensorflow.org/guide/data_performance#parallelizing_data_extraction
# todo use https://www.tensorflow.org/tutorials/distribute/input
dataset = dataset.prefetch(tf.data.AUTOTUNE)
dataset = dataset.batch(10)  # todo make batchsize configurable

# todo we need to export - as a callback?


prediction = modified_model.predict(dataset)

print(prediction["prediction"])
