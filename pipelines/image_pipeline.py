import abc
import base64
import os
from collections import Counter
from time import sleep

import imageio as iio
import numpy as np
import tensorflow as tf
from fastwarc.warc import ArchiveIterator

from helpers import get_file_stream, create_s3_client
from pipelines.generic_pipeline import Pipeline


class ImagePipeline(Pipeline, abc.ABC):

    def __init__(self, image_size, out_dir, max_content_length):
        self.image_size = image_size
        self.out_dir = out_dir
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)
        self.max_content_length = max_content_length

        super().__init__()

        def ragged_to_tensor(prediction, original_image, url):
            return prediction, original_image.to_tensor(), url

        self.dataset = self.dataset.map(ragged_to_tensor, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

    def get_dataset(self):
        return tf.data.Dataset.from_generator(self.driver_generator, output_signature=(
            tf.TensorSpec(shape=self.image_size + (3,), dtype=tf.float32),  # resized_image
            tf.RaggedTensorSpec(shape=(None, None, 3), dtype=tf.uint8, ragged_rank=2),  # original_image
            tf.TensorSpec(shape=(), dtype=tf.string)))  # url

    def get_distributed_filter(self):
        def distributed_filter(image):
            return True

        return distributed_filter

    def get_generator_factory(self):
        """
        return value is a generator that must not use any self.* attributes. Those must be copied to variables outside of the generator first #todo rework this description
        :return:
        """
        acc_counter = self.acc_counter
        image_size = self.image_size
        max_content_length = self.max_content_length
        distributed_filter = self.get_distributed_filter()
        BUCKET_NAME = self.BUCKET_NAME
        AWS_ACCESS_KEY_ID = self.AWS_ACCESS_KEY_ID
        AWS_SECRET = self.AWS_SECRET
        ENDPOINT_URL = self.ENDPOINT_URL
        acceptable_types = ['image/jpeg', 'image/gif', 'image/bmp', 'image/png']

        def generator_factory(file_name):
            s3_client = create_s3_client(AWS_ACCESS_KEY_ID, AWS_SECRET, ENDPOINT_URL)
            stream = get_file_stream(s3_client, BUCKET_NAME, file_name)
            for record in ArchiveIterator(stream, max_content_length=max_content_length):
                try:
                    if record.headers is None:
                        acc_counter.add(Counter({"n_record_headers_none": 1}))
                        continue
                    if record.http_headers is None:
                        acc_counter.add(Counter({"n_http_headers_none": 1}))
                        continue
                    if record.headers['WARC-Type'] == 'response' and record.content_length >= 128:
                        content_type = str(record.http_headers.get('Content-Type')).lower()
                        if content_type.startswith('image/') and any(
                                content_type.startswith(t) for t in acceptable_types):
                            url = str(record.headers['WARC-Target-URI'])
                            content = record.reader.read()
                            try:
                                image = tf.io.decode_image(content, channels=3, expand_animations=False)
                            except tf.errors.InvalidArgumentError:  # todo assess error rate
                                # todo maybe this is the problem? https://resiliparse.chatnoir.eu/en/stable/man/parse/http.html#read-chunked-http-payloads
                                acc_counter.add(Counter({"n_decoding_exception": 1}))
                                continue
                            if not distributed_filter(image):
                                acc_counter.add(Counter({"n_distributed_filter_not_passed": 1}))
                                continue
                            resized = tf.image.resize(tf.cast(image, tf.float32) / 255., image_size, antialias=True)
                            original_image = tf.RaggedTensor.from_tensor(image, ragged_rank=2)
                            yield resized, original_image, url
                            acc_counter.add(Counter({"n_node_results": 1}))
                            sleep(5)  # todo remove ????
                        else:
                            acc_counter.add(Counter({"n_wrong_content_type": 1}))
                    else:
                        acc_counter.add(Counter({"n_wrong_warc_type": 1}))
                except:
                    acc_counter.add(Counter({"n_unhandled_record_exceptions": 1}))
                    continue
            acc_counter.add(Counter({"n_finished_warc_files": 1}))

        return generator_factory

    def export(self, prediction, original_image, url):
        prediction = np.reshape(prediction, ())
        print(url.decode("utf-8"), prediction)
        iio.imwrite(f"{self.out_dir}/{base64.urlsafe_b64encode(url[:128]).decode('utf-8')}_{prediction:1.4f}.jpg",
                    original_image)
