FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
LABEL maintainer "tim.lee"

USER root

COPY ./docker/gpu_check_pytorch.py .
COPY ./docker/requirements.txt .
COPY ./docker/jupyter_notebook_config.py .

RUN apt-get update && apt-get install -y \
    git \
    wget \
    htop \
    screen \
    vim \
    curl \
    jq \
    tmux && apt-get clean all

# create another user called ML within the container, make it a sudoer + docker
RUN useradd ml
RUN usermod -aG sudo ml

USER ml

WORKDIR /home/ml
ENV HOME /home/ml


RUN pip install -U pip
RUN pip install -r requirements.txt

RUN jupyter notebook --generate-config
RUN jupyter nbextension enable --py widgetsnbextension

CMD jupyter lab --config jupyter_notebook_config.py
