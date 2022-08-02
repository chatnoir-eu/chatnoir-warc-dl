from pipelines.multimodal_pipeline import MultimodalPipeline
from pipelines.tools.passthrough_model import PassthroughModelPipeline


class MultimodalPairsExport(PassthroughModelPipeline, MultimodalPipeline):
    """
    Todo docstring
    """

    def __init__(self):
        image_size = (150, 150)  # rescale images to the format accepted by the model
        out_dir = "data/multimodal_pairs_export/out/"
        max_content_length = 4000000  # 4MB maximum image size
        super().__init__(image_size=image_size, out_dir=out_dir, max_content_length=max_content_length)


if __name__ == "__main__":
    p = MultimodalPairsExport()
    p.run()
