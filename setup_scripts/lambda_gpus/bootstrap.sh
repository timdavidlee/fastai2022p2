#!/bin/bash

set -e

colors=$(tput colors)
if (($colors >= 8)); then
  red='\033[0;31m'
  green='\033[0;32m'
  yellow='\033[1;33m'
  nocolor='\033[00m'
else
  red=
  green=
  nocolor=
fi

echo -e "${yellow}: adding a docker group if it doesn't exist"
sudo getent group || sudo groupadd docker

echo -e "${yellow}: adding the current user: [${USER}] to the docker group"
sudo usermod -aG docker $USER

echo -e "${yellow}: adjusting access to the docker socket information"
sudo chmod 666 /var/run/docker.sock

echo -e "${yellow}: updating the ubuntu system"
sudo apt-get update

echo -e "${yellow}: installing base system (not container) conveniences"
sudo apt-get install -y \
  wget \
  curl \
  htop \
  vim \
  nano \
  tmux \
  screen \
  jq

echo -e "${yellow}: verifying a gpu is available"
nvidia-smi

echo -e "${yellow}: creating lessons directory and cloning the lessons repos"
mkdir -p ~/ml_lessons
cd ~/ml_lessons
git clone https://github.com/fastai/course22p2
git clone https://github.com/fastai/diffusion-nbs


echo -e "${yellow}: installing nvidia-docker2"
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installation-guide
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# update and install
sudo apt-get update
sudo apt-get install -y nvidia-docker2

echo -e "${yellow}: restarting docker"
sudo systemctl restart docker

echo -e "${yellow}: testing with a simple container"
sudo docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi