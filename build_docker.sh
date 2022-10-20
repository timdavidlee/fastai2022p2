docker build --tag=fastai-gpu-jlab:local -f ./docker/Dockerfile .

echo "if the docker image was built successfully you should see a new entry:"
docker image ls