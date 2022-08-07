import base64
import os
from urllib.parse import urljoin

import imageio as iio
import tensorflow as tf
import torch
from fastwarc.warc import ArchiveIterator
from resiliparse.parse import detect_encoding
from resiliparse.parse.html import HTMLTree
from transformers import CLIPProcessor, CLIPModel

from helpers import get_file_stream, create_s3_client, build_index
from pipelines.pipeline import Pipeline


class ClipAltTexts(Pipeline):
    """
    This pipeline extracts images along with the alt tags that are used with them on html pages. Then, a CLIP filter
    is applied. The resulting dataset is similar to https://arxiv.org/abs/2111.02114. This pipeline works similar to
    MultimodalPipeline to extract the images from the WARC files. This also means that it currently only works with
    the Internet Archive WARC files.
    """

    def __init__(self):
        self.out_dir = "data/clip_alt_texts/out/"
        self.max_content_length = 4000000  # 4MB maximum image size

        super().__init__()

    def get_model(self):
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        return model

    def predict(self, model_inputs, *args):
        def predict_py(input_ids, attention_mask, pixel_values):
            model_inputs_pt = {"input_ids": torch.tensor(input_ids.numpy()[0], dtype=torch.int64),
                               "attention_mask": torch.tensor(attention_mask.numpy()[0], dtype=torch.int64),
                               "pixel_values": torch.tensor(pixel_values.numpy()[0], dtype=torch.float32)}
            prediction = float(self.model(**model_inputs_pt).logits_per_text[0, 0])
            return prediction

        prediction = tf.reshape(tf.py_function(predict_py, inp=model_inputs, Tout=tf.float64), [-1, ])
        return prediction, *args

    def batch(self, dataset, _):
        return dataset.batch(1)

    def get_signature(self):
        return (
            (tf.TensorSpec(shape=(1, None), dtype=tf.int32), tf.TensorSpec(shape=(1, None), dtype=tf.int32),
             tf.TensorSpec(shape=(1, 3, 224, 224), dtype=tf.float32)),  # clip inputs
            tf.TensorSpec(shape=(), dtype=tf.string),  # text_url
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),  # original_image
            tf.TensorSpec(shape=(), dtype=tf.string),  # alt_text
            tf.TensorSpec(shape=(), dtype=tf.string))  # image_url

    def get_distributed_image_filter(self):

        def distributed_filter(image):
            return True

        return distributed_filter

    def get_generator_factory(self):

        max_content_length = self.max_content_length
        distributed_image_filter = self.get_distributed_image_filter()
        AWS_ACCESS_KEY_ID = self.AWS_ACCESS_KEY_ID
        AWS_SECRET = self.AWS_SECRET
        ENDPOINT_URL = self.ENDPOINT_URL
        acceptable_types = ['image/jpeg', 'image/gif', 'image/bmp', 'image/png']

        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        def generator_factory(file_identifier):
            s3_client = create_s3_client(AWS_ACCESS_KEY_ID, AWS_SECRET, ENDPOINT_URL)
            index = None
            stream = get_file_stream(s3_client, file_identifier)
            for record in ArchiveIterator(stream, max_content_length=max_content_length):
                try:
                    if record.headers is None:
                        continue
                    if record.http_headers is None:
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
                                continue

                            if tree.body is None:
                                continue

                            if len(tree.body.get_elements_by_tag_name("img")) == 0:
                                continue
                            imgs = [(urljoin(url, img.getattr("src")), img.getattr("alt")) for img in
                                    tree.body.get_elements_by_tag_name("img") if img.getattr("alt")]
                            if len(imgs) == 0:
                                continue

                            if index is None:
                                try:
                                    index = build_index(s3_client, file_identifier)
                                except:
                                    continue

                            for img, alt_text in imgs:

                                if not img in index:
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
                                                continue
                                            content = record.reader.read()
                                            try:
                                                image = tf.io.decode_image(content, channels=3, expand_animations=False)
                                            except tf.errors.InvalidArgumentError:
                                                continue
                                            if not distributed_image_filter(image):
                                                continue

                                            inputs = processor(text=alt_text, images=image.numpy().transpose((2, 0, 1)),
                                                               return_tensors="tf", padding=True)

                                            yield tuple(inputs.values()), url, image, alt_text, image_url

                except:
                    continue

        return generator_factory

    def filter(self, prediction, *args):
        return tf.reshape(prediction >= 20, ())

    def export(self, prediction, text_url, original_image, alt_text, image_url):
        dirname = (f"{self.out_dir}/{base64.urlsafe_b64encode(text_url[:64]).decode('utf-8')}_"
                   f"{base64.urlsafe_b64encode(image_url[:64]).decode('utf-8')}")
        os.makedirs(dirname, exist_ok=True)
        with open(f"{dirname}/meta.txt", "w") as f:
            f.write(f"{prediction}\n")
            f.write(f"{text_url.decode('utf-8')}\n")
            f.write(f"{image_url.decode('utf-8')}\n")
            f.write(f"{alt_text.decode('utf-8')}\n")

        iio.imwrite(f"{dirname}/img.jpg", original_image)


if __name__ == "__main__":
    p = ClipAltTexts()
    p.run()
