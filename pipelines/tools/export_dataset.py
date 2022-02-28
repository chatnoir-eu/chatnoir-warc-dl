import abc
import os
import threading

import tensorflow as tf

from pipelines.generic_pipeline import Pipeline


class ExportDatasetPipeline(Pipeline, abc.ABC):
    def __init__(self, *args, dataset_export_dir, **kwargs):
        self.dataset_export_dir = dataset_export_dir
        os.makedirs(self.dataset_export_dir, exist_ok=True)
        super().__init__(*args, **kwargs)

    def run(self):
        self.t = threading.Thread(target=self.feed_executors, daemon=True)
        self.t.start()
        tf.data.experimental.save(self.dataset, self.dataset_export_dir)

    def export(self, *args):
        return
