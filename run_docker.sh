#!/bin/bash

echo "starting up a container in detached mode, running jupyterlab in the background"
docker run -it --rm -d --gpus all \
    --name gpu-jlab-pytorch \
    -v /home/ubuntu/data:/home/ml/data \
    -v /home/ubuntu/fastai2022p2:/home/ml/fastai2022p2 \
    -v /home/ubuntu/ml_lessons:/home/ml/ml_lessons \
    -v /home/ubuntu/.huggingface:/home/ml/.huggingface \
    -p 8888:8888 \
    -p 8889:8889 \
    -e TRANSFORMERS_CACHE=/home/ml/data/.cache/huggingface/ \
    fastai-gpu-jlab:local

docker ps | grep gpu-jlab-pytorch
