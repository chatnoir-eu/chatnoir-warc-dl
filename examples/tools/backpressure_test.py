import threading
import time

import numpy as np

from pipelines.tools.passthrough_model import PassthroughModelPipeline

SHAPE = (100,)  # data shape is so large that unbounded execution would lead to OOM very fast


class BackpressureTestPipeline(PassthroughModelPipeline):

    def get_bucket_files(self):
        return [f"file_{i}" for i in range(10)]

    def get_generator_factory(self):
        def generator_factory(file_name):
            for i in range(100):
                payload = np.empty(SHAPE, dtype=np.float32)
                descriptor = f"no {i} in {file_name}"
                time.sleep(.1)
                yield {"payload": payload, "descriptor": descriptor}

        return generator_factory

    def get_dataset(self):
        pass
        # return tf.data.Dataset.from_generator(self.driver_generator, output_signature=(
        #    tf.TensorSpec(shape=SHAPE, dtype=tf.float32),  # large data
        #    tf.TensorSpec(shape=(), dtype=tf.string)))  # descriptor

    def export(self, *args):
        print(*args)

    def start_threads(self):
        threading.Thread(target=self.feed_executors, daemon=True).start()

    def predict(self, model_input, *args):
        time.sleep(3)  # simulate slowness
        return super().predict(model_input, *args)


if __name__ == "__main__":
    p = BackpressureTestPipeline()
    p.run()
