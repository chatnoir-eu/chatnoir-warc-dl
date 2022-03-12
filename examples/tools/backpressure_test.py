import threading
import time

import numpy as np
import tensorflow as tf

from pipelines.tools.passthrough_model import PassthroughModelPipeline

SHAPE = (10000000,)  # data shape is so large that unbounded execution would lead to OOM very fast


class BackpressureTestPipeline(PassthroughModelPipeline):
    """
    This pipeline tests the ability of the node workers to wait for the driver to process data before continuing with
    the iteration and consuming more memory.
    """

    def get_bucket_files(self):
        return [f"file_{i}" for i in range(10)]

    def get_generator_factory(self):
        def generator_factory(file_identifier):
            for i in range(100):
                payload = np.empty(SHAPE, dtype=np.float32)
                descriptor = f"no {i} in {file_identifier}"
                time.sleep(.1)
                yield payload, descriptor

        return generator_factory

    def get_signature(self):
        return (
            tf.TensorSpec(shape=SHAPE, dtype=tf.float32),  # large payload data
            tf.TensorSpec(shape=(), dtype=tf.string))  # descriptor

    def export(self, *args):
        print(args[-1])

    def start_threads(self):
        threading.Thread(target=self.feed_executors, daemon=True).start()


if __name__ == "__main__":
    p = BackpressureTestPipeline()
    p.run()
