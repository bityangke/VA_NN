**Running VA_NN using the docker image**

##### Build docker

`docker build -t view_adaptive/pytorch1.2:jiaming_huang -f docker/Dockerfile`

##### Run docker

We only consider the gpu case, and you have to install nvidia-docker. You can reference to [github](https://github.com/NVIDIA/nvidia-docker)

`nvidia-docker run --name va_cnn -p 6006:6006 -v /home/hjm/docker:/workspace/VA_NN -it view_adaptive/pytorch1.2`
