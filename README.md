# web-archive-keras

This pipeline allows to extract data from WARC files on a CPU cluster and to stream it to a GPU server, where it is
processed. This allows to quickly retrieve data (text or images) from the WARC files that gets classified as positive by
a deep learning model.

Code for a simple image classification pipeline and a Huggingface text transformer pipeline is provided
in [examples](examples). It is easily adaptable to support other custom Keras models.

## General Functionality

The pipeline architecture can be described as follows:

![Pipeline architecture](docs/architecture.svg)

- Using PySpark, the WARC files are distributed among the CPU cluster workers.
- These workers use FastWARC to iterate over the records and apply a first CPU-based filter.
- The record streams get pickled and passed to the GPU server via TCP.
- The streams are converted into Tensorflow Datasets and interleaved into a single dataset structure.
- The batched dataset is passed through the Keras model.
- The classification results are used to filter the dataset.
- The final results are saved.

## Setup

### Requirements

To run the pipeline, the following hardware/software setup is required:

- GPU server running Python 3.8 (other versions might work as well), Tensorflow 2 with CUDA and a Hadoop instance
- CPU cluster with YARN
- WARC files in an S3 storage

For the GPU
server, [a Docker image is provided](https://github.com/niklasdeckers/web-archive-keras/pkgs/container/web-archive-keras)
.

Alternatively, you can install the required Python packages using

	pip3 install -r requirements.txt

### Configurations

Clone the repository.

Duplicate the `config-template.ini` as `config.ini` and make the appropriate changes. For tuning the `BATCHSIZE` and
number of `SPARK_INSTANCES` depending on your Keras model of choice, see [Profiling](#Profiling).

Modify the xml files inside the hadoop directory according to your cluster setup and provide them to the runner by
setting the `HADOOP_CONF_DIR` environment variable. You should also provide the `HADOOP_USER_NAME` environment variable
to identify your Hadoop jobs.

## Running the Examples

The pipeline must be run from the GPU server (or from within a Docker container running on the GPU server).

`cd` into the repository's toplevel directory. It is important to use the correct `PYTHONPATH`. You can run the image
classification example using

	PYTHONPATH=. HADOOP_CONF_DIR=./hadoop/ HADOOP_USER_NAME=$USER \
    python3 examples/meme_classifier/meme_classifier_pipeline.py

## Profiling

In order to tune the `BATCHSIZE` and number of `SPARK_INSTANCES` depending on the used Keras model, the used data and
the hardware/software configuration, the Tensorflow profiler should be used to estimate whether the CPU cluster or the
GPU server is the bottleneck. You will have to start with an initial guess for both values. A good starting point might
be a high `BATCHSIZE` so that your GPU will ideally be fully utilized with respect to the Keras model of your choice.

In the `config.ini`, set `enable_logging = yes`. When executing your pipeline the next time, a profiler log will be
created in the `data/logs` directory.

To view the logs, you need to start a tensorboard session with the plugin `tensorboard_plugin_profile` installed:

	pip3 install -U tensorboard_plugin_profile

From within the repository's toplevel directory, run

	tensorboard --logdir data/logs --host 0.0.0.0

Open the tensorboard in your browser. In the `Profile` mode, select a run and select the `tf_data_bottleneck_analysis`
tool. Take a look at the Input Pipeline Graph. If a `Generator` step is marked red, the CPU cluster is the bottleneck.
You should consider increasing the `SPARK_INSTANCES` in the config to allow more CPU workers in parallel.
If `ParallelMapV2` is marked red, the GPU cluster is the bottleneck. You could try to increase GPU resources, starting
with `BATCHSIZE`. If you can't handle the load from the CPU workers, you could consider decreasing the `SPARK_INSTANCES`
to free up resources.

## Adding Custom Keras Models

You can easily use custom Keras models for the classification. See the [example pipelines](examples) for image and text
based classifiers.

## Writing Code for the Nodes

When using custom Keras models, you might also want to write custom code for the CPU cluster. This code is responsible
for the extraction of data from the WARC files and the CPU-based preprocessing steps.

Please be aware that the custom code will be pickled by PySpark to be sent to the workers. This also means that you can
not use certain objects like `self` in the pipelines. Best practice is to copy all values that you need from `self` into
local variables and then defining a function that uses these values - either inside
the `Pipeline.get_generator_factory()` method itself or in the [`helpers.py`](helpers.py), from where you can safely use
imported methods as it is explicitly distributed to the workers.

If you would like to use additional Python packages on the workers, we
recommend [distributing a venv archive](https://spark.apache.org/docs/latest/api/python/user_guide/python_packaging.html#using-virtualenv)
. For the packages listed in the [`requirements.txt`](requirements.txt), there is a prebuilt archive available inside
the prebuilt Docker image (defined by the Dockerfile). The property in the `config.ini` for using this method
is `enable_prebuilt_dependencies = yes`.

## Planned Features

– Model training – Multi-GPU – Combining text and image information
