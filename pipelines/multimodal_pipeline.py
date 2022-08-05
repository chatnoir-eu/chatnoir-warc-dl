import abc
import base64
import os
from collections import Counter
from urllib.parse import urljoin

import imageio as iio
import numpy as np
import tensorflow as tf
from fastwarc.warc import ArchiveIterator
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse import detect_encoding
from resiliparse.parse.html import HTMLTree

from helpers import get_file_stream, create_s3_client, build_index
from pipelines.pipeline import Pipeline


class MultimodalPipeline(Pipeline, abc.ABC):
    """
    This pipeline extracts html files from the WARC files as well as images (urls + raw image data) that are
    referenced using img elements in the html files. This requires that there is an appropriately placed CDX index
    file in .cdx.gz format to allow locating the image. Currently, this is only the case for Internet Archive WARC
    files. It should not be expected that all images can be located. No attempts will be made by this pipeline to
    crawl for images online that could not be found. This pipeline streams the following to the driver/GPU: An
    (optionally tokenized) version of the website text, which should be as clean as possible (useful for neural
    network input), a version of the image that is resized to image_size and normalized to 1.0 (useful for neural
    network input), an original version of the text as a string, the website url, the original uint8 version of the
    image using a RaggedTensor format (variable image size) to allow batching, the image url.
    """

    def __init__(self, image_size, out_dir, max_content_length):
        self.image_size = image_size
        self.out_dir = out_dir
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)
        self.max_content_length = max_content_length

        super().__init__()

        def ragged_to_tensor(prediction, text_for_export, text_url, original_image, image_url):
            return prediction, text_for_export, text_url, original_image.to_tensor(), image_url

        self.dataset = self.dataset.map(ragged_to_tensor, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

    def get_signature(self):
        return (
            (self.get_tokens_spec(),  # text for classification
             tf.TensorSpec(shape=self.image_size + (3,), dtype=tf.float32)),  # resized_image
            tf.TensorSpec(shape=(), dtype=tf.string),  # text for export
            tf.TensorSpec(shape=(), dtype=tf.string),  # text_url
            tf.RaggedTensorSpec(shape=(None, None, 3), dtype=tf.uint8, ragged_rank=2),  # original_image
            tf.TensorSpec(shape=(), dtype=tf.string))  # image_url

    def get_distributed_text_filter(self):
        """
        Overridable method that provides a filter, which is executed on the pyspark cluster nodes.
        The returned distributed_filter must not use self. Needed attributes of self should be extracted into variables
        outside of the definition of distributed_filter, which may then use these variables.
        """

        def distributed_filter(text):
            return True

        return distributed_filter

    def get_distributed_image_filter(self):

        def distributed_filter(image):
            return True

        return distributed_filter

    def get_distributed_combined_filter(self):

        def distributed_filter(text, image):
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
        image_size = self.image_size
        max_content_length = self.max_content_length
        distributed_text_filter = self.get_distributed_text_filter()
        distributed_image_filter = self.get_distributed_image_filter()
        distributed_combined_filter = self.get_distributed_combined_filter()
        tokenizer = self.get_tokenizer()
        AWS_ACCESS_KEY_ID = self.AWS_ACCESS_KEY_ID
        AWS_SECRET = self.AWS_SECRET
        ENDPOINT_URL = self.ENDPOINT_URL
        acceptable_types = ['image/jpeg', 'image/gif', 'image/bmp', 'image/png']

        def generator_factory(file_identifier):
            s3_client = create_s3_client(AWS_ACCESS_KEY_ID, AWS_SECRET, ENDPOINT_URL)
            index = None
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
                        content_type = str(record.http_content_type).lower()
                        if content_type.startswith("text/html"):
                            url = str(record.headers['WARC-Target-URI'])
                            html_bytes = record.reader.read()
                            try:
                                encoding = record.http_charset
                                if encoding is None:
                                    encoding = detect_encoding(html_bytes)
                                tree = HTMLTree.parse_from_bytes(html_bytes, encoding)
                            except:
                                acc_counter.add(Counter({"n_parsing_exception": 1}))
                                continue

                            if tree.body is None:
                                acc_counter.add(Counter({"n_no_body": 1}))
                                continue
                            imgs = [urljoin(url, img.getattr("src")) for img in
                                    tree.body.get_elements_by_tag_name("img")]
                            if len(imgs) == 0:
                                acc_counter.add(Counter({"n_no_img_elements": 1}))
                                continue

                            prediction_text = extract_plain_text(tree, preserve_formatting=False,
                                                                 main_content=True, list_bullets=False,
                                                                 alt_texts=False, links=False,
                                                                 form_fields=False, noscript=False)

                            export_text = extract_plain_text(tree, preserve_formatting=True, main_content=True,
                                                             list_bullets=True, alt_texts=True, links=True,
                                                             form_fields=True, noscript=True)

                            if not distributed_text_filter(prediction_text):
                                acc_counter.add(Counter({"n_distributed_text_filter_not_passed": 1}))
                                continue

                            tokenized_prediction_text = tokenizer(prediction_text)

                            if index is None:
                                try:
                                    index = build_index(s3_client, file_identifier)
                                except:
                                    acc_counter.add(Counter({"n_index_file_exception": 1}))
                                    continue

                            for img in imgs:
                                if not img in index:
                                    acc_counter.add(Counter({"n_img_not_found": 1}))
                                    continue
                                warcfile, offset = index[img]
                                with get_file_stream(s3_client, warcfile,
                                                     range=f"bytes={offset}-{offset + max_content_length}") as image_stream:
                                    record = next(ArchiveIterator(image_stream, max_content_length=max_content_length))
                                    if record.headers['WARC-Type'] == 'response' and record.content_length >= 128:
                                        content_type = str(record.http_content_type).lower()
                                        if content_type.startswith('image/') and any(
                                                content_type.startswith(t) for t in acceptable_types):
                                            image_url = str(record.headers['WARC-Target-URI'])
                                            if image_url != img:
                                                acc_counter.add(Counter({"n_img_misplaced": 1}))
                                                continue
                                            content = record.reader.read()
                                            try:
                                                image = tf.io.decode_image(content, channels=3, expand_animations=False)
                                            except tf.errors.InvalidArgumentError:
                                                acc_counter.add(Counter({"n_decoding_exception": 1}))
                                                continue
                                            if not distributed_image_filter(image):
                                                acc_counter.add(Counter({"n_distributed_image_filter_not_passed": 1}))
                                                continue

                                            if not distributed_combined_filter(prediction_text, image):
                                                acc_counter.add(
                                                    Counter({"n_distributed_combined_filter_not_passed": 1}))
                                                continue

                                            resized = tf.image.resize(tf.cast(image, tf.float32) / 255., image_size,
                                                                      antialias=True)
                                            original_image = tf.RaggedTensor.from_tensor(image, ragged_rank=2)

                                            yield (tokenized_prediction_text,
                                                   resized), export_text, url, original_image, image_url

                                            acc_counter.add(Counter({"n_node_results": 1}))
                                        else:
                                            acc_counter.add(Counter({"n_img_wrong_content_type": 1}))
                                    else:
                                        acc_counter.add(Counter({"n_img_wrong_warc_type": 1}))

                        else:
                            acc_counter.add(Counter({"n_wrong_content_type": 1}))
                    else:
                        acc_counter.add(Counter({"n_wrong_warc_type": 1}))
                except:
                    acc_counter.add(Counter({"n_unhandled_record_exceptions": 1}))
                    continue
            acc_counter.add(Counter({"n_finished_warc_files": 1}))

        return generator_factory

    def export(self, prediction, text_for_export, text_url, original_image, image_url):
        prediction = np.reshape(prediction, ())
        print(text_url.decode("utf-8"), image_url.decode("utf-8"), prediction)
        dirname = (f"{self.out_dir}/{base64.urlsafe_b64encode(text_url[:64]).decode('utf-8')}_"
                   f"{base64.urlsafe_b64encode(image_url[:64]).decode('utf-8')}_{prediction:1.4f}")
        os.makedirs(dirname, exist_ok=True)
        with open(f"{dirname}/{base64.urlsafe_b64encode(text_url[:128]).decode('utf-8')}_{prediction:1.4f}.txt",
                  "w") as f:
            f.write(text_for_export.decode("utf-8"))
        iio.imwrite(f"{dirname}/{base64.urlsafe_b64encode(image_url[:128]).decode('utf-8')}_{prediction:1.4f}.jpg",
                    original_image)
