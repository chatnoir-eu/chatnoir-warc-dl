import collections

import boto3
from pyspark import AccumulatorParam


def create_s3_client(AWS_ACCESS_KEY_ID, AWS_SECRET, ENDPOINT_URL):
    session = boto3.session.Session(AWS_ACCESS_KEY_ID, AWS_SECRET)
    return session.client(
        service_name='s3',
        endpoint_url=ENDPOINT_URL,
    )


def get_file_stream(s3_client, bucket, key):
    response = s3_client.get_object(
        Bucket=bucket,
        Key=key
    )
    return response['Body']._raw_stream


class CounterAccumulatorParam(AccumulatorParam):
    def zero(self, v):
        return collections.Counter()

    def addInPlace(self, acc1, acc2):
        return acc1 + acc2

