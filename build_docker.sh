echo "cleaning up any dangling images"
docker system prune -f

echo "if the docker image was built successfully you should see a new entry:"
docker build --tag=fastai-gpu-jlab:local -f ./docker/Dockerfile .

echo "pruning any accidental incomplete builds"
docker system prune -f

echo "if the docker image was built successfully you should see a new entry:"
docker image ls