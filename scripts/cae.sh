#!/bin/bash

docker run \
  --name cae \
  --rm \
  --detach \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --shm-size 32gb \
  --volume $(pwd):/workspace \
  ad_project \
  python src/main.py \
  --models CAE \
  --lr 1e-3
