import resiliparse.parse.lang
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
        max_content_length = 4000000
        super().__init__(out_dir=out_dir, max_content_length=max_content_length)

    def get_model(self):
        model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english",
                                                                     cache_dir="models/hatespeech_classifier/")
        return model

    def predict(self, model_input, *args):
        prediction, *_ = super().predict(model_input)
        logits = prediction["logits"]
        probabilities = tf.nn.softmax(logits)
        return probabilities[:, 0], *args  # extract NEGATIVE classification result for whole batch

    def get_tokens_spec(self):
        return {'input_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                'attention_mask': tf.TensorSpec(shape=(None,), dtype=tf.int32)}

    def batch(self, dataset, batchsize):
        return dataset.padded_batch(batchsize)

    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english",
                                                  cache_dir="models/hatespeech_classifier/")

        def tokenizer_func(inp):
            return tokenizer(inp).data

        return tokenizer_func

    def get_distributed_filter(self):
        def distributed_filter(text):
            if len(text) < 1000:  # only extract long texts
                return False
            return resiliparse.parse.lang.detect_fast(text)[0] == "en"  # only extract english texts

        return distributed_filter

    def filter(self, prediction, *args):
        return tf.reshape(prediction > .9, ())


if __name__ == "__main__":
    p = HatespeechClassifierPipeline()
    p.run()
