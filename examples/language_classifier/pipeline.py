import os
import urllib.request

import tensorflow as tf
from tensorflow import keras

from pipelines.text_pipeline import TextPipeline


class LanguageClassifierPipeline(
    TextPipeline):  # todo we need another example as language classification can also be handled on betaweb directly
    """
    This is an example text classification pipeline based on
    https://huggingface.co/papluca/xlm-roberta-base-language-detection.
    Here, we use this model to extract German (de) website text.
    """

    def __init__(self):
        out_dir = "data/language_classifier/out/"
        max_content_length = 4000000  # todo define proper text limit
        super().__init__(out_dir=out_dir, max_content_length=max_content_length)

        def multiple_to_one(prediction, export_text, url):
            return prediction[4], export_text, url  # extract German (de) classification result

        self.dataset = self.dataset.map(multiple_to_one, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

    def get_model(self):
        model_source = "https://huggingface.co/papluca/xlm-roberta-base-language-detection/resolve/main/tf_model.h5"
        model_file = "models/language_classifier/tf_model.h5"
        if not os.path.isfile(model_file):
            os.makedirs(os.path.dirname(model_file), exist_ok=True)
            print("Downloading model...")
            urllib.request.urlretrieve(model_source, model_file)
            print("Model download finished.")
        additional_source = "https://huggingface.co/papluca/xlm-roberta-base-language-detection/resolve/main/config.json"
        model = keras.models.load_model(model_file)
        return model

    def get_distributed_filter(self):
        def distributed_filter(text):
            return len(text) > 1000  # only extract long texts

        return distributed_filter

    def filter(self, prediction, *args):
        return tf.reshape(prediction[4] > .9, ())  # extract German (de) classification result


if __name__ == "__main__":
    p = LanguageClassifierPipeline()
    p.run()
