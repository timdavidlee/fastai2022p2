FROM pytorch/pytorch:latest
LABEL maintainer "tim.lee"

USER root

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
RUN useradd -m -d /home/ml ml
RUN usermod -aG sudo ml

USER ml

WORKDIR /home/ml
ENV HOME /home/ml

COPY ./docker/gpu_check_pytorch.py .
COPY ./docker/requirements.txt .
COPY ./docker/jupyter_notebook_config.py .

ENV VIRTUAL_ENV=/home/ml/fastai_venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install -U pip
RUN pip install -r requirements.txt

RUN jupyter nbextension enable --py widgetsnbextension

CMD jupyter lab --config jupyter_notebook_config.py
