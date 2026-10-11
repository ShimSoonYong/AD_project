#!/bin/bash

docker run \
  --name ad_proj \
  --rm \
  --detach \
  --gpus all \
  --ipc host \
  --volume $(pwd):/workspace \
  ad_project \
  python src/main.py \
  --models Autoencoder CNN \
  --epochs 20
