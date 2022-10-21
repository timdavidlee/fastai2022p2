FROM  nvidia/cuda:11.6.1-devel-ubuntu20.04
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
    bash-completion \
    tmux && apt-get clean all

# install python
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y \
    software-properties-common \
    && apt install python3.10

# symlink to python3
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3

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

# install specific version of pytorch
RUN pip install \
    torch==1.12.0+cu116 \
    torchvision==0.13.0+cu116 \
    torchaudio==0.12.0 \
    --extra-index-url https://download.pytorch.org/whl/cu116

# install related course libs
RUN pip install -r requirements.txt

RUN jupyter nbextension enable --py widgetsnbextension

CMD jupyter lab --config jupyter_notebook_config.py
