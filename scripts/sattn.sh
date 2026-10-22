#!/bin/bash

docker run \
  --name sattn \
  --rm \
  --detach \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --shm-size 32gb \
  --volume $(pwd):/workspace \
  ad_project \
  python src/main.py \
  --models SAttn \
  --threshold 0.1 \
  --lr 1e-4
