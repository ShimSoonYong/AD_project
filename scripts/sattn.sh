#!/bin/bash

seeds=("42" "123" "951" "503" "9874")

for seed in "${seeds[@]}"; do
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
    --seed "$seed" \
    --lr 1e-5

  docker wait sattn
done
