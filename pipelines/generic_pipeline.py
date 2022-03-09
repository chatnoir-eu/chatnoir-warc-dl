import abc
import collections
import configparser
import json
import os
import pickle
import socket
import threading
import time
from queue import Queue

import tensorflow as tf
from pyspark import SparkContext, SparkConf

from helpers import create_s3_client, CounterAccumulatorParam


class Pipeline(abc.ABC):
    def __init__(self):

        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

        self.BUCKET_NAMES = json.loads(self.config["s3"]["BUCKET_NAMES"])
        self.AWS_ACCESS_KEY_ID = self.config["s3"]["AWS_ACCESS_KEY_ID"]
        self.AWS_SECRET = self.config["s3"]["AWS_SECRET"]
        self.ENDPOINT_URL = self.config["s3"]["ENDPOINT_URL"]

        conf = SparkConf()
        conf_list = [("spark.executor.instances", str(self.config["pyspark"]["SPARK_INSTANCES"]))]
        if self.config.getboolean("pyspark", "enable_prebuilt_dependencies"):
            # deploy prebuilt dependencies according to
            # https://spark.apache.org/docs/latest/api/python/user_guide/python_packaging.html#using-virtualenv
            os.environ['PYSPARK_PYTHON'] = "./environment/bin/python"
            conf_list.append(("spark.yarn.dist.archives", "/pyspark_venv.tar.gz#environment"))
        conf.setAll(conf_list)
        self.sc = SparkContext(master="yarn", appName="web-archive-keras", conf=conf)
        self.sc.addPyFile("helpers.py")

        self.acc_counter = self.sc.accumulator(collections.Counter(), CounterAccumulatorParam())

        self.BATCHSIZE = int(self.config["tensorflow"]["BATCHSIZE"])

        self.model = self.get_model()

        self.q = Queue()

        self.dataset = self.complete_ds(int(self.config["pyspark"]["SPARK_INSTANCES"]))
        self.dataset = self.dataset.prefetch(tf.data.AUTOTUNE)
        self.dataset = self.batch(self.dataset, self.BATCHSIZE)

        self.dataset = self.dataset.map(self.predict, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

        self.dataset = self.dataset.unbatch()

        self.dataset = self.dataset.filter(self.filter)

    @abc.abstractmethod
    def get_model(self):
        pass
    @abc.abstractmethod
    def get_signature(self):
        pass

    def complete_ds(self, n_instances):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        self.HOST = socket.gethostname()
        self.PORT = s.getsockname()[1]
        s.listen()

        def server():
            while True:
                conn, _ = s.accept()
                infile = conn.makefile(mode="rb")
                self.q.put(infile)

        threading.Thread(target=server, daemon=True).start()

        base_ds = tf.data.Dataset.range(n_instances)

        def gen(q):
            while True:
                f = q.get()
                if f is None:
                    q.put(None)
                    return
                while True:
                    try:
                        yield pickle.load(f)
                    except EOFError:
                        f.close()
                        break

        def ds_from_queue(q, signature):
            ds = tf.data.Dataset.from_generator(lambda: gen(q), output_signature=signature)
            # maybe another prefetch is helpful here?
            return ds

        interleaved_ds = base_ds.interleave(lambda _: ds_from_queue(self.q, self.get_signature()),
                                            num_parallel_calls=tf.data.AUTOTUNE,
                                            deterministic=False,
                                            cycle_length=n_instances)

        return interleaved_ds

    def batch(self, dataset, batchsize):
        return dataset.batch(batchsize, drop_remainder=True)

    def start_threads(self):
        threading.Thread(target=self.feed_executors, daemon=True).start()

        def print_stats():
            while True:
                time.sleep(10)
                print("accumulator:", self.acc_counter)

        threading.Thread(target=print_stats, daemon=True).start()

        def profiler():
            options = tf.profiler.experimental.ProfilerOptions(host_tracer_level=3,
                                                               python_tracer_level=1,
                                                               device_tracer_level=1,
                                                               delay_ms=int(self.config.getfloat("profiler",
                                                                                                 "logging_delay_s") * 1000))
            os.makedirs('./data/logs/', exist_ok=True)
            tf.profiler.experimental.start('./data/logs/', options=options)
            time.sleep(self.config.getfloat("profiler", "logging_delay_s") \
                       + self.config.getfloat("profiler", "logging_duration_s"))
            tf.profiler.experimental.stop()

        if self.config.getboolean("profiler", "enable_logging"):
            threading.Thread(target=profiler).start()

    def run(self):
        self.start_threads()
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
        filenames = []
        for BUCKET_NAME in self.BUCKET_NAMES:
            s3_client = create_s3_client(self.AWS_ACCESS_KEY_ID, self.AWS_SECRET, self.ENDPOINT_URL)
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=BUCKET_NAME)
            filenames += [(BUCKET_NAME, obj['Key']) for page in pages for obj in page['Contents']]
        return filenames

    def feed_executors(self):
        files = self.get_bucket_files()
        rdd = self.sc.parallelize(files, len(files))
        generator_factory = self.get_generator_factory()
        HOST, PORT = self.HOST, self.PORT

        def node_client(generator, HOST, PORT):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((HOST, PORT))
                with s.makefile(mode="wb") as outfile:
                    for record in generator:
                        pickle.dump(record, outfile)

        rdd.foreach(lambda file_identifier: node_client(generator_factory(file_identifier), HOST, PORT))
        self.q.put(None)

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
