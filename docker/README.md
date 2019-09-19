**Running VA_NN using the docker image**

##### Build docker

`docker build -t view_adaptive/pytorch1.2:jiaming_huang -f Dockerfile .`

##### Run docker

We only consider the gpu case, and you have to install nvidia-docker. You can reference to [github](https://github.com/NVIDIA/nvidia-docker)

`docker run --name va_cnn -p 6006:6006 -v .../data/NTU-RGB+D:/workspace/data/NTU-RGB+D -v .../weight:/workspace/weights -it view_adaptive/pytorch1.2`
