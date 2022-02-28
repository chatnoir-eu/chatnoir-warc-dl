import abc

import tensorflow as tf

from pipelines.generic_pipeline import Pipeline


class PassthroughModelPipeline(Pipeline, abc.ABC):

    def get_model(self):
        return None

    def predict(self, *args):
        return args

    @tf.function
    def filter(self, *args):
        return True
