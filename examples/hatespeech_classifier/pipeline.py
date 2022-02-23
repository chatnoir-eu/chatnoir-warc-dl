import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

from pipelines.text_pipeline import TextPipeline


class HatespeechClassifierPipeline(TextPipeline):
    """
    This is an example text classification pipeline based on
    https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english.
    Here, we use this model to extract hatespeech.
    """

    def __init__(self):
        out_dir = "data/hatespeech_classifier/out/"
        max_content_length = 4000000  # todo define proper text limit
        super().__init__(out_dir=out_dir, max_content_length=max_content_length)

        def multiple_to_one(prediction, export_text, url):
            return prediction[0], export_text, url  # extract NEGATIVE classification result

        self.dataset = self.dataset.map(multiple_to_one, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

    def get_model(self):
        model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english",
                                                                     cache_dir="models/hatespeech_classifier/")
        return model

    def get_tokens_spec(self):
        return {'input_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                # todo use raggedtensor and allow masked batching
                'attention_mask': tf.TensorSpec(shape=(None,), dtype=tf.int32)}

    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english",
                                                  cache_dir="models/hatespeech_classifier/")

        def tokenizer_func(inp):
            return tokenizer(inp).data

        return tokenizer_func

    def get_distributed_filter(self):
        def distributed_filter(text):
            return len(text) > 1000  # only extract long texts

        return distributed_filter

    def filter(self, prediction, *args):
        return tf.reshape(prediction["logits"][0] > .9, ())  # extract NEGATIVE classification result


if __name__ == "__main__":
    p = HatespeechClassifierPipeline()
    p.run()
