import abc
import configparser
import threading

import tensorflow as tf
from pyspark import SparkContext, SparkConf

from helpers import NonPicklableQueue, ResultsParam, create_s3_client


class Pipeline(abc.ABC):
    def __init__(self):

        config = configparser.ConfigParser()
        config.read('config.ini')

        self.BUCKET_NAME = config["s3"]["BUCKET_NAME"]
        self.AWS_ACCESS_KEY_ID = config["s3"]["AWS_ACCESS_KEY_ID"]
        self.AWS_SECRET = config["s3"]["AWS_SECRET"]
        self.ENDPOINT_URL = config["s3"]["ENDPOINT_URL"]

        conf = SparkConf()
        conf.setAll([("spark.executor.instances", str(config["pyspark"]["SPARK_INSTANCES"]))])
        self.sc = SparkContext(master="yarn", appName="spark-test", conf=conf)
        self.sc.addPyFile("helpers.py")

        self.q = NonPicklableQueue()
        self.acc = self.sc.accumulator([], ResultsParam(self.q))

        self.BATCHSIZE = int(config["tensorflow"]["BATCHSIZE"])

        self.model = self.get_model()
        self.dataset = self.get_dataset()
        self.dataset = self.dataset.prefetch(tf.data.AUTOTUNE)
        self.dataset = self.dataset.batch(self.BATCHSIZE)
        # todo allow padded_batch
        # todo does padded_batch also work with ragged images?

        self.dataset = self.dataset.map(self.predict, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

        self.dataset = self.dataset.unbatch()

        self.dataset = self.dataset.filter(self.filter)

    @abc.abstractmethod
    def get_model(self):
        pass

    def run(self):
        self.t = threading.Thread(target=self.feed_executors, daemon=True)
        self.t.start()
        for data in self.dataset.as_numpy_iterator():
            self.export(*data)

    @abc.abstractmethod
    def get_generator_factory(self):
        """
        return value is a generator that must not use any self.* attributes. Those must be copied to variables outside of the generator first#todo rework this description
        :return:
        """
        pass

    def get_bucket_files(self):
        s3_client = create_s3_client(self.AWS_ACCESS_KEY_ID, self.AWS_SECRET, self.ENDPOINT_URL)
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.BUCKET_NAME)
        return [obj['Key'] for page in pages for obj in page['Contents']]

    def feed_executors(self):
        files = self.get_bucket_files()
        rdd = self.sc.parallelize(files, len(files))
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
        """
        For the prediction, the model input is expected to be at the first position in the dataset structure
        :param model_input:
        :param args:
        :return:
        """
        prediction = self.model(model_input, training=False)
        return prediction, *args

    @abc.abstractmethod
    def filter(self, *args):
        pass

    @abc.abstractmethod
    def export(self, *args):
        pass
