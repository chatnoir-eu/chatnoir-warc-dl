import collections
import gzip
import os

import boto3
from pyspark import AccumulatorParam


def create_s3_client(AWS_ACCESS_KEY_ID, AWS_SECRET, ENDPOINT_URL):
    session = boto3.session.Session(AWS_ACCESS_KEY_ID, AWS_SECRET)
    return session.client(
        service_name='s3',
        endpoint_url=ENDPOINT_URL,
    )


def get_file_stream(s3_client, file_identifier, range=None):
    bucket, key = file_identifier
    if range is None:
        response = s3_client.get_object(
            Bucket=bucket,
            Key=key
        )
    else:
        response = s3_client.get_object(
            Bucket=bucket,
            Key=key,
            Range=range
        )
    return response['Body']._raw_stream


def build_index(s3_client, file_identifier):
    bucket, key = file_identifier
    dirname = os.path.dirname(key)
    key = os.path.join(dirname, os.path.basename(dirname) + ".cdx.gz")
    with get_file_stream(s3_client, (bucket, key)) as raw_index_stream:
        with gzip.open(raw_index_stream) as index_stream:
            columns = next(index_stream).decode("utf-8").strip().split()
            assert columns[0] == "CDX"
            del columns[0]
            url_column = columns.index("a")
            warcfile_column = columns.index("g")
            offset_column = columns.index("V")
            mimetype_column = columns.index("m")
            index = dict()  # should contain values ((bucket,key),offset)
            for line in index_stream:
                splitted = line.decode("utf-8").strip().split(" ")
                if splitted[mimetype_column] == "warc/revisit":
                    continue
                index[splitted[url_column]] = (
                (bucket, os.path.join(os.path.dirname(dirname), splitted[warcfile_column])),
                int(splitted[offset_column]))
    return index


class CounterAccumulatorParam(AccumulatorParam):
    def zero(self, v):
        return collections.Counter()

    def addInPlace(self, acc1, acc2):
        return acc1 + acc2
