#!/bin/bash

set -e

# ensure that an IP or instance is provided
if [ $# -ne 1 ]; then
  echo "Usage $(basename $0) <target-host>"
  exit 1
fi

TARGET_HOST=$1
USER=ubuntu  # this is default for lambda GPU

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

echo -e "${yellow}[${TARGET_HOST}]: adding a docker group if it doesn't exist"
ssh -T $USER@$TARGET_HOST <<'ENDSSH'
sudo getent group || sudo groupadd docker
ENDSSH

echo -e "${yellow}[${TARGET_HOST}]: adding the current user: [${USER}] to the docker group"
ssh -T $USER@$TARGET_HOST <<'ENDSSH'
sudo usermod -aG docker $USER
ENDSSH

echo -e "${yellow}[${TARGET_HOST}]: updating the ubuntu system"
ssh -T $USER@$TARGET_HOST <<'ENDSSH'
sudo apt-get update
ENDSSH

echo -e "${yellow}[${TARGET_HOST}]: installing some convenience linux libs"
ssh -T $USER@$TARGET_HOST <<'ENDSSH'
sudo apt-get install -y \
  wget \
  curl \
  htop \
  vim \
  nano \
  tmux \
  screen \
  jq
ENDSSH


echo -e "${yellow}[${TARGET_HOST}]: ensure that the GPU is available and what GPU version is running"
ssh -T $USER@$TARGET_HOST <<'ENDSSH'
nvidia-smi
ENDSSH
