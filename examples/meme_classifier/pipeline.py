from tensorflow import keras

from pipelines.image_pipeline import ImagePipeline


class MemePipeline(ImagePipeline):
    def load_model(self):
        model = keras.models.load_model(
            "models/chollet.h5")  # https://github.com/samon11/meme-classifier/blob/master/chollet.h5
        return model


if __name__ == "__main__":
    p = MemePipeline()
    p.run()
