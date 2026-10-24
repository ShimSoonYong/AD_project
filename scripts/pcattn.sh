#!/bin/bash

docker run \
  --name pcattn \
  --rm \
  --detach \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --shm-size 32gb \
  --volume $(pwd):/workspace \
  ad_project \
  python src/main.py \
  --models PCAttn \
  --lr 2e-4
