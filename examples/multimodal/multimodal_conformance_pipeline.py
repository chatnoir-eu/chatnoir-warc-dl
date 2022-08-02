from pipelines.multimodal_pipeline import MultimodalPipeline
from pipelines.tools.passthrough_model import PassthroughModelPipeline


class MultimodalConformancePipeline(PassthroughModelPipeline, MultimodalPipeline):
    """
    Todo docstring
    """

    def __init__(self):
        image_size = (150, 150)  # rescale images to the format accepted by the model
        out_dir = "data/multimodal_conformance/out/"
        max_content_length = 4000000  # 4MB maximum image size
        super().__init__(image_size=image_size, out_dir=out_dir, max_content_length=max_content_length)

    # def get_model(self):#todo
    #     model_source = "https://github.com/samon11/meme-classifier/raw/master/chollet.h5"
    #     model_file = "models/meme_classifier/chollet.h5"
    #     if not os.path.isfile(model_file):
    #         os.makedirs(os.path.dirname(model_file), exist_ok=True)
    #         print("Downloading model...")
    #         urllib.request.urlretrieve(model_source, model_file)
    #         print("Model download finished.")
    #     model = keras.models.load_model(model_file)
    #     return model

    # def get_distributed_image_filter(self):#todo
    #     def distributed_filter(image):
    #         size_x, size_y, *_ = image.shape
    #         return 150 <= size_x <= 1000 and 150 <= size_y <= 1000  # typical meme image size
    #
    #     return distributed_filter
    #
    # def filter(self, prediction, *args):#todo
    #     return tf.reshape(prediction > .95, ())


if __name__ == "__main__":
    p = MultimodalConformancePipeline()
    p.run()
