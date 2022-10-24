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
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt install -y python3.9 python3.9-venv

# symlink to python3
RUN ln -sf /usr/bin/python3.9 /usr/bin/python3

# create another user called ML within the container, make it a sudoer + docker
RUN useradd -m -d /home/ml ml
RUN usermod -aG sudo ml

USER ml

WORKDIR /home/ml
ENV HOME /home/ml

# this is where hugging face will save temp files. we wiant in the mapped directory
# so we don't need to restart all the time, this should be made globally outside of the docker
ENV TRANSFORMERS_CACHE /home/ml/data/.cache/huggingface/transformers

COPY ./docker/gpu_check_pytorch.py .
COPY ./docker/requirements.txt .
COPY ./docker/jupyter_notebook_config.py .

ENV VIRTUAL_ENV=/home/ml/fastai_venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# upgrading the pytorch version to pytorch:1.12.0-11.3, defaults to 1.11.1:10.2 for some reason
RUN pip install -U pip

# install related course libs
RUN pip install -r --no-cache-dir requirements.txt

# install torch afterwards to up-version
RUN pip install --no-cache-dir \
    torch==1.12.0+cu113 \
    torchvision==0.13.0+cu113 \
    torchaudio==0.12.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113

RUN jupyter nbextension enable --py widgetsnbextension

CMD jupyter lab --config jupyter_notebook_config.py
