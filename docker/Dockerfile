FROM nvidia/cuda:10.1-runtime-ubuntu18.04
MAINTAINER jiamingNo1 jiaming19.huang@foxmail.com

RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        cmake \
        git \
        curl \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

EXPOSE 6006
WORKDIR /workspace

RUN git clone https://github.com/jiamingNo1/VA_NN.git

RUN pip3 install --upgrade pip && \
    for req in $(cat requirements.txt); do pip3 install $req; done
