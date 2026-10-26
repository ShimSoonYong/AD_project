#!/bin/bash

seeds=("42")
thresholds=("0.95" "0.90" "0.85" "0.8" "0.75")

for seed in "${seeds[@]}"; do
  for threshold in "${thresholds[@]}"; do
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
      --threshold_p "$threshold" \
      --seed "$seed" \
      --lr 1e-5

    docker wait pcr

    docker rm -f pcr

  done
done
