#!/bin/bash

docker run \
  --name fmemae \
  --detach \
  --rm \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --shm-size 32gb \
  --volume "$(pwd)":/workspace \
  ad_project \
  python src/main.py \
  --models FeatureMemAE \
  --mode inference
