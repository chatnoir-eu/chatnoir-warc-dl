import abc
import base64
import os
from time import sleep

import numpy as np
import tensorflow as tf
from fastwarc.warc import ArchiveIterator
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse import detect_encoding
from resiliparse.parse.html import HTMLTree
from resiliparse.process_guard import time_guard, mem_guard, MemoryLimitExceeded, ExecutionTimeout

from helpers import create_s3_client, get_file_stream
from pipelines.generic_pipeline import Pipeline


class TextPipeline(Pipeline, abc.ABC):

    def __init__(self, out_dir, max_content_length):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.max_content_length = max_content_length

        super().__init__()

    def get_dataset(self):
        return tf.data.Dataset.from_generator(self.driver_generator, output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),  # text for classification
            tf.TensorSpec(shape=(), dtype=tf.string),  # text for export
            tf.TensorSpec(shape=(), dtype=tf.string)))  # url

    def get_distributed_filter(self):
        def distributed_filter(text):
            return True

        return distributed_filter

    def get_generator_factory(self):
        """
        return value is a generator that must not use any self.* attributes. Those must be copied to variables outside of the generator first #todo rework this description
        :return:
        """
        max_content_length = self.max_content_length
        distributed_filter = self.get_distributed_filter()
        BUCKET_NAME = self.BUCKET_NAME
        AWS_ACCESS_KEY_ID = self.AWS_ACCESS_KEY_ID
        AWS_SECRET = self.AWS_SECRET
        ENDPOINT_URL = self.ENDPOINT_URL

        def generator_factory(file_name):
            s3_client = create_s3_client(AWS_ACCESS_KEY_ID, AWS_SECRET, ENDPOINT_URL)
            stream = get_file_stream(s3_client, BUCKET_NAME, file_name)
            for record in ArchiveIterator(stream, max_content_length=max_content_length):
                try:
                    if record.headers is None:
                        continue
                    if record.http_headers is None:
                        continue
                    if record.headers['WARC-Type'] == 'response' and record.content_length >= 128:
                        content_type = str(record.http_headers.get('Content-Type')).lower()
                        if content_type.startswith("text/html"):
                            url = str(record.headers['WARC-Target-URI'])
                            try:
                                with time_guard(timeout=10) as t_guard, mem_guard(max_memory=1024 * 50, grace_period=2):
                                    html_bytes = record.reader.read()
                                    try:
                                        encoding = None
                                        for p in content_type.split(';'):
                                            p = p.strip()
                                            if p.startswith('charset='):
                                                encoding = p[8:].lower()
                                                break
                                        if encoding is None:
                                            encoding = detect_encoding(html_bytes)
                                        tree = HTMLTree.parse_from_bytes(html_bytes, encoding)
                                    except:
                                        continue

                                    prediction_text = extract_plain_text(tree, preserve_formatting=False,
                                                                         main_content=True, list_bullets=False,
                                                                         alt_texts=False, links=False,
                                                                         form_fields=False, noscript=False)

                                    export_text = extract_plain_text(tree, preserve_formatting=True, main_content=True,
                                                                     list_bullets=True, alt_texts=True, links=True,
                                                                     form_fields=True, noscript=True)

                                    if not distributed_filter(prediction_text):
                                        continue

                                    yield prediction_text, export_text, url

                            except (ExecutionTimeout, MemoryLimitExceeded):
                                continue
                            sleep(5)  # todo remove ????
                except:
                    raise  # todo better: continue

        return generator_factory

    def export(self, prediction, export_text, url):
        prediction = np.reshape(prediction, ())
        print(url.decode("utf-8"), prediction)
        with open(f"{self.out_dir}/{base64.urlsafe_b64encode(url).decode('utf-8')}_{prediction:1.4f}.txt") as f:
            f.write(export_text)
