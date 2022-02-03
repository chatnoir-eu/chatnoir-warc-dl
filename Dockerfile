FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
	software-properties-common default-jre curl

RUN curl https://dlcdn.apache.org/hadoop/common/hadoop-2.10.1/hadoop-2.10.1.tar.gz --output hadoop-2.10.1.tar.gz
RUN tar -zxvf hadoop-2.10.1.tar.gz
RUN cp -a hadoop-2.10.1/. /

COPY hadoop /etc/hadoop/

RUN add-apt-repository ppa:deadsnakes/ppa

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
	python3.8 \
    curl \
    python3.8-distutils

# set python3.8 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

RUN curl -Ss https://bootstrap.pypa.io/get-pip.py | python3

RUN python3 -m pip install --upgrade pip && python3 -m pip install setuptools --no-cache-dir

COPY requirements.txt .

RUN python3 -m pip install -r requirements.txt --no-cache-dir
