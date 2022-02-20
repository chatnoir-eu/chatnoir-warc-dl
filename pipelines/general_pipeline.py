import abc
import threading

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

        self.model = self.get_model()
        self.dataset = self.get_dataset()
        self.dataset = self.dataset.prefetch(tf.data.AUTOTUNE)
        self.dataset = self.dataset.batch(self.BATCHSIZE)

        self.dataset = self.dataset.map(self.predict, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

        self.dataset = self.dataset.unbatch()

        self.dataset = self.dataset.filter(self.filter)

    @abc.abstractmethod
    def get_model(self):
        pass

    def run(self):
        self.t = threading.Thread(target=self.target, daemon=True)
        self.t.start()
        for data in self.dataset.as_numpy_iterator():
            self.export(*data)

    @abc.abstractmethod
    def get_generator_factory(self):
        """
        return value is a generator that must not use any self.* attributes. Those must be copied to variables outside of the generator first
        :return:
        """
        pass

    def target(self):
        rdd = self.sc.parallelize(range(32), 32)
        acc = self.acc
        rdd.flatMap(self.get_generator_factory()).foreach(lambda x: acc.add([x]))
        self.q.put(None)

    def driver_generator(self):
        while True:
            elem = self.q.get()
            if elem is None:
                return
            yield elem

    @abc.abstractmethod
    def get_dataset(self):
        pass

    def predict(self, model_input, *args):
        prediction = self.model(model_input, training=False)
        return prediction, *args

    @abc.abstractmethod
    def filter(self, *args):
        pass

    @abc.abstractmethod
    def export(self, *args):
        pass
