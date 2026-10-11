#!/bin/bash

docker run \
  --name=ad_proj \
  --rm \
  --detach \
  --gpus=all \
  --ipc=host \
  --memory=64gb \
  --volume=$(pwd):/workspace \
  ad_project \
  python src/main.py
