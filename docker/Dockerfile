FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
LABEL maintainer "tim.lee"

WORKDIR /home/ml
ENV HOME /home/ml

RUN apt-get update && apt-get install -y \
    git \
    wget \
    htop \
    screen \
    vim \
    curl \
    jq \
    tmux

RUN pip install -U \
    fastai \
    diffusers \
    transformers \
    huggingface-hub \
    jupyter-lab \
    notebook

CMD jupyter lab --ip=0.0.0.0 --no-browser --notebook-dir=/home/ml/
