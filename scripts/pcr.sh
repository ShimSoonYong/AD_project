#!/bin/bash

seeds=("42" "123" "951" "503" "9874")

for seed in "${seeds[@]}"; do
  docker run \
    --name pcr \
    --rm \
    --detach \
    --gpus all \
    --ipc host \
    --ulimit memlock=-1 \
    --shm-size 32gb \
    --volume $(pwd):/workspace \
    ad_project \
    python src/main.py \
    --models PCR \
    --seed "$seed" \
    --lr 1e-5

  docker wait pcr
done
