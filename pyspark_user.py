import threading
import time
from queue import Queue

from pyspark import SparkContext, TaskContext, AccumulatorParam, SparkConf

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


import socket


def oneToMultiple(i):
    t = ([100] * 3 + [1] * 29)[i]
    # Add some delay
    time.sleep(t / 10)
    yield (i, t, socket.gethostname())


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


for each in gen():
    print(each)
