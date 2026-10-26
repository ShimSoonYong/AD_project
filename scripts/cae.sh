#!/bin/bash

seeds=("42" "123" "951" "503" "9874")

for seed in "${seeds[@]}"; do
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
    --seed "$seed" \
    --lr 5e-5

  docker wait cae
done
