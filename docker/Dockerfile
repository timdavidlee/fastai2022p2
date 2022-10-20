FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
LABEL maintainer "tim.lee"


USER ubuntu

WORKDIR /home/ubuntu
ENV HOME /home/ubuntu

COPY ./docker/gpu_check_pytorch.py .
COPY ./docker/requirements.txt .
COPY ./docker/jupyter_notebook_config.py .

RUN sudo apt-get update && sudo apt-get install -y \
    git \
    wget \
    htop \
    screen \
    vim \
    curl \
    jq \
    tmux

RUN pip install -U pip
RUN pip install -r requirements.txt

RUN jupyter notebook --generate-config
RUN jupyter nbextension enable --py widgetsnbextension

CMD jupyter lab --config jupyter_notebook_config.py
