#!/bin/bash

docker run \
  --name ae \
  --rm \
  --detach \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --shm-size 32gb \
  --volume $(pwd):/workspace \
  ad_project \
  python src/main.py \
  --models AE \
  --lr 1e-4
