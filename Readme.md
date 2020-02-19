# Instructions to run to set everything up:

## Install Docker and Nvidia, build the containers

Machine Learning work should be done in a container. To install ["nvidia-docker2"](<https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0)#prerequisites>).
Here's how to do it:

### Install docker

```
#For Ubuntu 18.04
#https://docs.docker.com/install/linux/docker-ce/ubuntu/
apt-get install apt-transport-https ca-certificates gnupg-agent software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

#sudo groupadd docker
sudo usermod -aG docker $USER

#Enable docker to run at startup
sudo systemctl enable docker

#Verify that Docker CE is installed correctly by running the hello-world image.
docker run hello-world
```

### Install nvidia-docker2

```
#https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update

sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd

#Test
docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
```

### Working with Container

```
# Build container, this command will build container locally (note the ".")
 # The container will build tensorflow with native CPU flags to get all instructions working natively on the build machine.
 # NOTE: these containers occupy a lot of space, 10's of GBs (or even towards 100 GB), so ensure there is sufficient free space

# cd to Dockerfiles folder
nvidia-docker build -f Dockerfile.td_base -t td_base .
nvidia-docker build -f Dockerfile.td_base_nvidia_ml -t td_base_nvidia_ml .
nvidia-docker build -f Dockerfile.td_train -t td_train .

#To run the new interactive container
 #Use --privileged if access needed to OTHER devices like USB drives.


 # NOTE: -v flag and the text that follows allows the user to mount the specified directory so it can be easily used by the docker container

# Run the container
nvidia-docker run --runtime=nvidia -it --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v /opt/path/to/fast/storage:/location/in/the/docker td_train


#Inside the container, run a few simple tests like:

import torch
torch.cuda.is_available()
```

## Inference - Run get_response.py

```
# All that is left is to inference, it logs the results to trained_model folder

# cd to your folder with the data
sudo python3 get_response.py
```
