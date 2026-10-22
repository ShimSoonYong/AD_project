#!/bin/bash

docker run \
  --name convsumae \
  --rm \
  --detach \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --shm-size 32gb \
  --volume $(pwd):/workspace \
  ad_project \
  python src/main.py \
  --models ConvSumAE \
  --threshold 0.1 \
  --lr 2e-5
