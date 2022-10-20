FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
LABEL maintainer "tim.lee"

USER root

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

USER ml

RUN pip install -U \
    fastai \
    diffusers \
    transformers \
    huggingface-hub \
    notebook

CMD jupyter notebook --ip=0.0.0.0 --no-browser --notebook-dir=/home/ml/ --Server.app.password='sha1:bfa0e68cf625678f6438e11d35577c23277d3dd8'
