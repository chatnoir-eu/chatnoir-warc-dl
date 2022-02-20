import abc
import base64
from time import sleep

import imageio as iio
import numpy as np
import requests
import tensorflow as tf

from pipelines.general_pipeline import Pipeline


class ImagePipeline(Pipeline, abc.ABC):
    @property
    @abc.abstractmethod
    def size(self):
        pass

    def __init__(self):
        super().__init__()

        def ragged_to_tensor(prediction, original_image, url):
            return prediction, original_image.to_tensor(), url

        self.dataset = self.dataset.map(ragged_to_tensor, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

    def get_dataset(self):
        return tf.data.Dataset.from_generator(self.driver_generator, output_signature=(
            tf.TensorSpec(shape=self.size + (3,), dtype=tf.float32),  # resized_image
            tf.RaggedTensorSpec(shape=(None, None, 3), dtype=tf.uint8, ragged_rank=2),  # original_image
            tf.TensorSpec(shape=(), dtype=tf.string)))  # url

    def get_generator_factory(self):
        """
        return value is a generator that must not use any self.* attributes. Those must be copied to variables outside of the generator first
        :return:
        """
        size = self.size

        def generator_factory(i):
            def get_result(url, size):
                r = requests.get(url, allow_redirects=True)
                image = tf.io.decode_image(r.content, channels=3, expand_animations=False)
                sleep(5)  # simulate IO slowness
                resized = tf.image.resize(tf.cast(image, tf.float32) / 255., size, antialias=True)
                original_image = tf.RaggedTensor.from_tensor(image, ragged_rank=2)
                return resized, original_image, url

            for ending in ["png", "jpg"]:
                url = f"https://www2.informatik.hu-berlin.de/~deckersn/data/test{i % 5}.{ending}"
                yield get_result(url, size)

        return generator_factory

    def export(self, prediction, original_image, url):
        prediction = np.reshape(prediction, ())
        print(url.decode("utf-8"), prediction)
        iio.imwrite(f"data/out/{base64.b64encode(url).decode('utf-8')}_{prediction:1.4f}.jpg", original_image)
