import tensorflow as tf
from tensorflow import keras

from pipelines.image_pipeline import ImagePipeline


class MemePipeline(ImagePipeline):
    def get_model(self):
        model = keras.models.load_model(
            "models/chollet.h5")  # https://github.com/samon11/meme-classifier/blob/master/chollet.h5
        return model

    def filter(self, prediction, *args):
        return tf.reshape(prediction > .9, ())


if __name__ == "__main__":
    p = MemePipeline()
    p.run()
