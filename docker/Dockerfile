FROM  pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
LABEL maintainer "tim.lee"

USER root

RUN apt-get update && apt-get install -y \
    apt-utils \
    sudo \
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
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get install -y \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev && apt-get clean all

# create another user called ML within the container, make it a sudoer + docker
RUN useradd -m -d /home/ml ml
RUN usermod -aG sudo ml

USER ml

WORKDIR /home/ml
ENV HOME /home/ml
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# get pyenv
RUN curl https://pyenv.run | bash

# the version of python should always be compared to pytorch to ensure compatibility
RUN pyenv install 3.9.15
RUN pyenv global 3.9.15
RUN pyenv rehash

# this is where hugging face will save temp files. we wiant in the mapped directory
# so we don't need to restart all the time, this should be made globally outside of the docker
ENV TRANSFORMERS_CACHE /home/ml/data/.cache/huggingface/transformers

COPY ./docker/gpu_check_pytorch.py .
COPY ./docker/requirements.txt .
COPY ./docker/jupyter_notebook_config.py .

# upgrading the pytorch version to pytorch:1.12.0-11.3, defaults to 1.11.1:10.2 for some reason
RUN pip install -U pip

# install related course libs
RUN pip install --no-cache-dir -r requirements.txt

# install torch afterwards to up-version
RUN pip install --no-cache-dir \
    torch==1.12.0+cu113 \
    torchvision==0.13.0+cu113 \
    torchaudio==0.12.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113

RUN jupyter nbextension enable --py widgetsnbextension

CMD jupyter lab --config jupyter_notebook_config.py
