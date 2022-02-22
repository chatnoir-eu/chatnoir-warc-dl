import os
import urllib.request

import tensorflow as tf
from tensorflow import keras

from pipelines.image_pipeline import ImagePipeline


class MemePipeline(ImagePipeline):
    """
    This is an example picture classifier pipeline based on https://github.com/samon11/meme-classifier.
    Due to the different training image distribution, it will produce extremely noisy results and thus should
    only serve demonstration purposes.
    """


    def get_model(self):
        model_source = "https://github.com/samon11/meme-classifier/raw/master/chollet.h5"
        model_file = "models/meme_classifier/chollet.h5"
        if not os.path.isfile(model_file):
            os.makedirs(os.path.dirname(model_file), exist_ok=True)
            print("Downloading model...")
            urllib.request.urlretrieve(model_source, model_file)
            print("Model download finished.")
        model = keras.models.load_model(model_file)
        return model

    def filter(self, prediction, *args):
        return tf.reshape(prediction > .9, ())


if __name__ == "__main__":
    image_size = (150, 150)
    out_dir = "data/meme_classifier/out/"
    p = MemePipeline(image_size, out_dir)
    p.run()
