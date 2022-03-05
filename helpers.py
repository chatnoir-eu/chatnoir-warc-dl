import collections
from queue import Queue

import boto3
from pyspark import TaskContext, AccumulatorParam


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


class ResultsParam(AccumulatorParam):
    def __init__(self, q):
        self.q = q
        super().__init__()

    def zero(self, v):
        return []

    def addInPlace(self, acc1, acc2):
        # This is executed on the workers so we have to
        # merge the results
        if (TaskContext.get() is not None and
                TaskContext().get().partitionId() is not None):
            acc1.extend(acc2)
            return acc1
        else:
            # This is executed on the driver so we discard the results
            # and publish to self instead
            assert len(acc1) == 0
            for x in acc2:
                self.q.put(x)
            return []


class CounterAccumulatorParam(AccumulatorParam):
    def zero(self, v):
        return collections.Counter()

    def addInPlace(self, acc1, acc2):
        return acc1 + acc2


class NonPicklableQueue(Queue):  # todo remove unused helpers
    def __getstate__(self):
        return None
