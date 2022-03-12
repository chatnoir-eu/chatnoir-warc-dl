import abc
import os

import tensorflow as tf

from pipelines.pipeline import Pipeline


class ExportDatasetPipeline(Pipeline, abc.ABC):
    """
    This pipeline is used to directly store datasets (as a result from the GPU computation step) using a sharded
    format provided by tensorflow.
    """

    def __init__(self, *args, dataset_export_dir=None, **kwargs):
        self.dataset_export_dir = dataset_export_dir
        os.makedirs(self.dataset_export_dir, exist_ok=True)
        super().__init__(*args, **kwargs)

    def run(self):
        self.start_threads()
        tf.data.experimental.save(self.dataset, self.dataset_export_dir)

    def export(self, *args):
        return
