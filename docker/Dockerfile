FROM nvidia/cuda:10.1-devel-ubuntu18.04
MAINTAINER jiamingNo1 jiaming19.huang@foxmail.com

RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        cmake \
        git \
        curl \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
EXPOSE 6006

RUN git clone https://github.com/jiamingNo1/VA_NN.git . && \
    pip3 install setuptools && \
    pip3 install -r requirements.txt
