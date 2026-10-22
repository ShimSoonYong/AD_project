#!/bin/bash

docker run \
  --name fmemae \
  --rm \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --shm-size 32gb \
  --volume $(pwd):/workspace \
  ad_project \
  python src/main.py \
  --models FeatureMemAE \
  --threshold 0.04 \
  --train_criterion SSIMLoss \
  --score_criterion SSIMLoss \
  --optimizer AdamW \
  --compactness_weight 0.0 \
  --separateness_weight 0.0 \
  --memory_margin 0.2 \
  --entropy_weight 1.3e-6 \
  --weight_decay 1e-4 \
  --latent_dim 64 \
  --hidden_dim 256 \
  --memory_size 100 \
  --temperature 1.0 \
  --shrink_threshold 0.01 \
  --patience 80 \
  --lr 0.001 \
  --epochs 400 \
  --augment none \
  --batch_size 64 \
  --num_workers 12 \
  --seed 42
