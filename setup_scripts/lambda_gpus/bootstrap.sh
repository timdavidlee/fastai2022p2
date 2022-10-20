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