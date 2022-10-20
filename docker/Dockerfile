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

RUN pip install -U --no-cache-dir\
    pip \
    fastai \
    diffusers \
    transformers \
    huggingface-hub \
    notebook

RUN jupyter notebook --generate-config
RUN jupyter nbextension enable --py widgetnbextension

CMD jupyter lab --ip=0.0.0.0 --no-browser --notebook-dir=/home/ml/ --Server.app.password='argon2:$argon2id$v=19$m=10240,t=10,p=8$Tb7raEKjGNxkyoFI558tgg$slnFXEftATF8bwdt8mvXWXMg7BQn5S3VAWXmxQs4KMM'
