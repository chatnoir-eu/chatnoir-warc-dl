import abc
import base64
import os
from collections import Counter

import numpy as np
import tensorflow as tf
from fastwarc.warc import ArchiveIterator
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse import detect_encoding
from resiliparse.parse.html import HTMLTree
from resiliparse.process_guard import time_guard, MemoryLimitExceeded, ExecutionTimeout

from helpers import create_s3_client, get_file_stream
from pipelines.pipeline import Pipeline


class TextPipeline(Pipeline, abc.ABC):
    """
    This pipeline extracts texts from websites from the WARC files. It streams the following to the driver/GPU:
    An (optionally tokenized) version of the website text, which should be as clean as possible (useful for neural
    network input),
    a original version of the text as a string,
    the website url.
    """

    def __init__(self, out_dir, max_content_length):
        self.out_dir = out_dir
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)
        self.max_content_length = max_content_length

        super().__init__()

    def get_signature(self):
        return (
            self.get_tokens_spec(),  # text for classification
            tf.TensorSpec(shape=(), dtype=tf.string),  # text for export
            tf.TensorSpec(shape=(), dtype=tf.string))  # url

    def get_distributed_filter(self):
        """
        Overridable method that provides a filter, which is executed on the pyspark cluster nodes.
        The returned distributed_filter must not use self. Needed attributes of self should be extracted into variables
        outside of the definition of distributed_filter, which may then use these variables.
        """

        def distributed_filter(text):
            return True

        return distributed_filter

    def get_tokens_spec(self):
        """
        Overridable method that returns a tf.TensorSpec which corresponds to the values returned by the tokenizer
        defined in get_tokenizer().
        """

        return tf.TensorSpec(shape=(), dtype=tf.string)

    def get_tokenizer(self):
        """
        Overridable method that provides a tokenizer, which is executed on the pyspark cluster nodes.
        The returned tokenizer must not use self. Needed attributes of self should be extracted into variables
        outside of the definition of tokenizer, which may then use these variables.
        """

        def tokenizer(text):
            return text

        return tokenizer

    def get_generator_factory(self):
        acc_counter = self.acc_counter
        max_content_length = self.max_content_length
        distributed_filter = self.get_distributed_filter()
        tokenizer = self.get_tokenizer()
        AWS_ACCESS_KEY_ID = self.AWS_ACCESS_KEY_ID
        AWS_SECRET = self.AWS_SECRET
        ENDPOINT_URL = self.ENDPOINT_URL

        def generator_factory(file_identifier):
            s3_client = create_s3_client(AWS_ACCESS_KEY_ID, AWS_SECRET, ENDPOINT_URL)
            stream = get_file_stream(s3_client, file_identifier)
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
                        if content_type.startswith("text/html"):
                            url = str(record.headers['WARC-Target-URI'])
                            try:
                                with time_guard(
                                        timeout=10):  # , mem_guard(max_memory=1024 * 50, grace_period=2):# todo add back again https://github.com/chatnoir-eu/chatnoir-resiliparse/blob/4f0b3bf7168228c947107bbe459d09d3923fa93e/resiliparse/resiliparse/process_guard.pyx#L75
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
                                        acc_counter.add(Counter({"n_decoding_exception": 1}))
                                        continue

                                    prediction_text = extract_plain_text(tree, preserve_formatting=False,
                                                                         main_content=True, list_bullets=False,
                                                                         alt_texts=False, links=False,
                                                                         form_fields=False, noscript=False)

                                    export_text = extract_plain_text(tree, preserve_formatting=True, main_content=True,
                                                                     list_bullets=True, alt_texts=True, links=True,
                                                                     form_fields=True, noscript=True)

                                    if not distributed_filter(prediction_text):
                                        acc_counter.add(Counter({"n_distributed_filter_not_passed": 1}))
                                        continue

                            except (ExecutionTimeout, MemoryLimitExceeded):
                                acc_counter.add(Counter({"n_resiliparse_guard_exceptions": 1}))
                                continue

                            yield tokenizer(prediction_text), export_text, url
                            acc_counter.add(Counter({"n_node_results": 1}))

                        else:
                            acc_counter.add(Counter({"n_wrong_content_type": 1}))
                    else:
                        acc_counter.add(Counter({"n_wrong_warc_type": 1}))
                except:
                    acc_counter.add(Counter({"n_unhandled_record_exceptions": 1}))
                    continue
            acc_counter.add(Counter({"n_finished_warc_files": 1}))

        return generator_factory

    def export(self, prediction, export_text, url):
        prediction = np.reshape(prediction, ())
        print(url.decode("utf-8"), prediction)
        with open(f"{self.out_dir}/{base64.urlsafe_b64encode(url[:128]).decode('utf-8')}_{prediction:1.4f}.txt",
                  "w") as f:
            f.write(export_text.decode("utf-8"))
