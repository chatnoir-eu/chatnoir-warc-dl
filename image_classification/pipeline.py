import base64
import threading
from queue import Queue
from time import sleep

import imageio as iio
import requests
import tensorflow as tf
from pyspark import SparkContext, TaskContext, AccumulatorParam, SparkConf
from tensorflow import keras

conf = SparkConf()
conf.setAll([("spark.executor.instances", "5")])
sc = SparkContext(master="yarn", appName="spark-test", conf=conf)


class ResultsParam(AccumulatorParam):
    def zero(self, v):
        return []

    def addInPlace(self, acc1, acc2):
        # This is executed on the workers so we have to
        # merge the results
        if (TaskContext.get() is not None and
                TaskContext().get().partitionId() is not None):
            acc1.extend(acc2)
            return acc1
        else:
            # This is executed on the driver so we discard the results
            # and publish to self instead
            assert len(acc1) == 0
            for x in acc2:
                q.put(x)
            return []


# Define accumulator
acc = sc.accumulator([], ResultsParam())


class NonPicklableQueue(Queue):
    def __getstate__(self):
        return None


q = NonPicklableQueue()


def process(x):
    global acc
    result = x
    acc.add([result])
    return result


BATCHSIZE = 3

size = (150, 150)


def get_result(filename):
    r = requests.get(filename, allow_redirects=True)
    image = tf.io.decode_image(r.content, channels=3, expand_animations=False)
    sleep(5)  # simulate IO slowness
    resized = tf.image.resize(tf.cast(image, tf.float32) / 255., size, antialias=True)
    return tf.RaggedTensor.from_tensor(image, ragged_rank=2), resized, filename


def oneToMultiple(i):
    for ending in ["png", "jpg"]:
        filename = f"https://www2.informatik.hu-berlin.de/~deckersn/data/test{i % 5}.{ending}"
        yield get_result(filename)


def target():
    rdd = sc.parallelize(range(32), 32)
    rdd.flatMap(oneToMultiple).foreach(process)
    q.put(None)


t = threading.Thread(target=target, daemon=True)
t.start()


def gen():
    while True:
        elem = q.get()
        if elem is None:
            return
        yield elem


model = keras.models.load_model(
    "models/chollet.h5")  # https://github.com/samon11/meme-classifier/blob/master/chollet.h5


def ds():
    return tf.data.Dataset.from_generator(lambda: gen(), output_signature=(
        tf.RaggedTensorSpec(shape=(None, None, 3), dtype=tf.uint8, ragged_rank=2),
        tf.TensorSpec(shape=size + (3,), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.string)))


dataset = ds()
dataset = dataset.prefetch(tf.data.AUTOTUNE)
dataset = dataset.batch(BATCHSIZE)


def predict(image, resized, url):
    prediction = model(resized, training=False)
    return prediction, image, url


dataset = dataset.map(predict, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

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
