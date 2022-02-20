import abc
import base64
import threading
from time import sleep

import imageio as iio
import requests
import tensorflow as tf
from pyspark import SparkContext, SparkConf

from helpers import NonPicklableQueue, ResultsParam


class Pipeline(abc.ABC):
    def __init__(self):
        conf = SparkConf()
        conf.setAll([("spark.executor.instances", "5")])
        self.sc = SparkContext(master="yarn", appName="spark-test", conf=conf)
        self.sc.addPyFile("helpers.py")

        self.q = NonPicklableQueue()
        self.acc = self.sc.accumulator([], ResultsParam(self.q))

        self.BATCHSIZE = 3

        self.size = (150, 150)

        self.model = self.load_model()
        self.dataset = self.ds()
        self.dataset = self.dataset.prefetch(tf.data.AUTOTUNE)
        self.dataset = self.dataset.batch(self.BATCHSIZE)

        self.dataset = self.dataset.map(self.predict, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

        self.dataset = self.dataset.unbatch()

        self.dataset = self.dataset.filter(self.filter)

        self.dataset = self.dataset.map(self.ragged_to_tensor, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

    @abc.abstractmethod
    def load_model(self):
        pass

    def run(self):
        self.t = threading.Thread(target=self.target, daemon=True)
        self.t.start()

        for data in self.dataset.as_numpy_iterator():
            self.export(*data)

    def oneToMultipleFactory(self):
        """
        return value is a generator that must not use any self.* attributes. Those must be copied to variables outside of the generator first
        :return:
        """
        size = self.size

        def oneToMultiple(i):
            def get_result(filename, size):
                r = requests.get(filename, allow_redirects=True)
                image = tf.io.decode_image(r.content, channels=3, expand_animations=False)
                sleep(5)  # simulate IO slowness
                resized = tf.image.resize(tf.cast(image, tf.float32) / 255., size, antialias=True)
                return tf.RaggedTensor.from_tensor(image, ragged_rank=2), resized, filename

            for ending in ["png", "jpg"]:
                filename = f"https://www2.informatik.hu-berlin.de/~deckersn/data/test{i % 5}.{ending}"
                yield get_result(filename, size)

        return oneToMultiple

    def target(self):
        rdd = self.sc.parallelize(range(32), 32)
        acc = self.acc
        rdd.flatMap(self.oneToMultipleFactory()).foreach(lambda x: acc.add([x]))
        self.q.put(None)

    def gen(self):
        while True:
            elem = self.q.get()
            if elem is None:
                return
            yield elem

    def ds(self):
        return tf.data.Dataset.from_generator(lambda: self.gen(), output_signature=(
            tf.RaggedTensorSpec(shape=(None, None, 3), dtype=tf.uint8, ragged_rank=2),
            tf.TensorSpec(shape=self.size + (3,), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.string)))

    def predict(self, image, resized, url):
        prediction = self.model(resized, training=False)
        return prediction, image, url

    def filter(self, prediction, image, url):
        return tf.reshape(prediction > .9, ())

    def ragged_to_tensor(self, prediction, image, url):
        return prediction, image.to_tensor(), url

    def export(self, prediction, image, url):
        prediction = prediction[0]
        print(url.decode("utf-8"), prediction)
        iio.imwrite(f"data/out/{base64.b64encode(url).decode('utf-8')}_{prediction:1.4f}.jpg", image)
