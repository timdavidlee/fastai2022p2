#!/bin/bash

set -e

# ensure that an IP or instance is provided
if [ $# -ne 1 ]; then
  echo "Usage $(basename $0) <target-host>"
  exit 1
fi

USER=ubuntu
TARGET=$1

ssh $USER@$TARGET -L 8888:localhost:8888