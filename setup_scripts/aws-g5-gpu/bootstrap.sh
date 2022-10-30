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

echo -e "${yellow}: install docker"
sudo apt update -y && sudo apt-get install -y docker.io

echo -e "${yellow}: adding a docker group if it doesn't exist"
sudo getent group docker || sudo groupadd docker

echo -e "${yellow}: adding the current user: [${USER}] to the docker group"
sudo usermod -aG docker $USER

# echo -e "${yellow}: adjusting access to the docker socket information"
# sudo chmod 666 /var/run/docker.sock

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

echo -e "${yellow}: checking that a gpu is installed"
sudo lspci -v | less | grep 'NVIDIA Corporation'

echo -e "${yellow}: ==========================================="
echo -e "${yellow}: will install cuda-11.6"
echo -e "${yellow}: 1. update linux"
sudo apt-get install linux-headers-$(uname -r)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.0-1_all.deb -O cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-drivers

echo -e "${yellow}: adding cuda to PATH"
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

echo -e "${yellow}: verify the version once its running"
nvidia-smi

echo -e "${yellow}: installing nvidia-docker2"
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installation-guide
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# update and install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo apt-get install -y nvidia-docker2

echo -e "${yellow}: restarting docker"
sudo systemctl restart docker

echo -e "${yellow}: testing with a simple container"
sudo docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi

echo -e "${yellow}: making data dir: /home/ubuntu/data, and huggingface cache directory"
mkdir -p /home/ubuntu/data
mkdir -p /home/ubuntu/data/.cache
mkdir -p /home/ubuntu/data/.cache/huggingface
mkdir -p /home/ubuntu/data/.cache/huggingface/transformers

echo -e "${yellow}: creating lessons directory and cloning (or pulling) the lessons repos"
mkdir -p ~/ml_lessons
cd ~/ml_lessons
git -C course22p2 pull || git clone https://github.com/fastai/course22p2
git -C diffusion-nbs pull || git clone https://github.com/fastai/diffusion-nbs
