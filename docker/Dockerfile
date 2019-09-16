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

RUN python3 -m pip install --upgrade pip && \
    pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install setuptools

RUN git clone https://github.com/jiamingNo1/VA_NN.git . && \
    pip3 install -r requirements.txt
