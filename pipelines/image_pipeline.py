import abc
import base64
from time import sleep

import imageio as iio
import requests
import tensorflow as tf

from pipelines.general_pipeline import Pipeline


class ImagePipeline(Pipeline, abc.ABC):

    def __init__(self):
        super().__init__()

        def ragged_to_tensor(prediction, image, url):
            return prediction, image.to_tensor(), url

        self.dataset = self.dataset.map(ragged_to_tensor, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

    def get_dataset(self):
        return tf.data.Dataset.from_generator(self.generator, output_signature=(
            tf.RaggedTensorSpec(shape=(None, None, 3), dtype=tf.uint8, ragged_rank=2),
            tf.TensorSpec(shape=self.size + (3,), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.string)))

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

    def filter(self, prediction, image, url):
        return tf.reshape(prediction > .9, ())

    def export(self, prediction, image, url):
        prediction = prediction[0]
        print(url.decode("utf-8"), prediction)
        iio.imwrite(f"data/out/{base64.b64encode(url).decode('utf-8')}_{prediction:1.4f}.jpg", image)
