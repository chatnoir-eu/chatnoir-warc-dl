import abc

import tensorflow as tf

from pipelines.generic_pipeline import Pipeline


class PassthroughModelPipeline(Pipeline, abc.ABC):
    """
    This pipeline replaces the Keras model functionality with a passthrough model.
    All dataset records will receive a prediction of 1.0 and all records will pass the filter on the driver.
    """

    def get_model(self):
        return None

    def predict(self, model_input, *args):
        return tf.ones((self.BATCHSIZE,)), *args

    @tf.function
    def filter(self, *args):
        return True
