#!/bin/bash

seeds=("42")
thresholds=("0.95" "0.90" "0.85" "0.8" "0.75")

for seed in "${seeds[@]}"; do
  for threshold in "${thresholds[@]}"; do
    docker run \
      --name pccae \
      --rm \
      --detach \
      --gpus all \
      --ipc host \
      --ulimit memlock=-1 \
      --shm-size 32gb \
      --volume $(pwd):/workspace \
      ad_project \
      python src/main.py \
      --models PCCAE \
      --threshold_p "$threshold" \
      --seed "$seed" \
      --lr 2e-4

    docker wait pccae

    docker rm -f pccae

  done
done
