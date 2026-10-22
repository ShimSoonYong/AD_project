#!/bin/bash

docker run \
  --name pcconvsumae \
  --rm \
  --detach \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --shm-size 32gb \
  --volume $(pwd):/workspace \
  ad_project \
  python src/main.py \
  --models PCConvSumAE \
  --threshold 0.04 \
  --lr 3e-4
