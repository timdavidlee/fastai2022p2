# this will be our data directory
mkdir -p /home/ubuntu/data

docker run -it --rm \
    -v /home/ubuntu/data:/home/ml/data \
    -v /home/ubuntu/fastai2022p2:/home/ml/fastai2022p2 \
    -p 8888:8888 \
    -p 8889:8889 \
    fastai-gpu-jlab:local
