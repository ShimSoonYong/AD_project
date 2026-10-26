#!/bin/bash

lr_options=("0.001" "0.0005" "0.0001" "0.0005" "0.00001")
criterons=("MSELoss" "L1Loss")
memory_sizes=("50" "100" "200")
latent_dims=("32" "64" "128")
thresholds=("0.95" "0.9" "0.85" "0.8" "0.75")
seeds=("42")

for seed in "${seeds[@]}"; do
  for lr in "${lr_options[@]}"; do
    for threshold in "${thresholds[@]}"; do
      for mem in "${memory_sizes[@]}"; do
        for lat in "${latent_dims[@]}"; do
          for cr in "${criterons[@]}"; do
            echo "=== Running with criterion=$cr, lr=$lr, threshold_p=$threshold, memory_size=$mem, latent_dim=$lat ==="

            docker run \
              --name fvae \
              --rm \
              --detach \
              --gpus all \
              --ipc host \
              --ulimit memlock=-1 \
              --shm-size 32gb \
              --volume "$(pwd)":/workspace \
              ad_project \
              python src/main.py \
              --models FeatureVAE \
              --train_criterion "$cr" \
              --score_criterion "$cr" \
              --optimizer AdamW \
              --compactness_weight 0.0 \
              --separateness_weight 0.0 \
              --memory_margin 0.2 \
              --entropy_weight 1.3e-6 \
              --weight_decay 1e-4 \
              --latent_dim "$lat" \
              --hidden_dim 256 \
              --memory_size "$mem" \
              --temperature 1.0 \
              --shrink_threshold 0.01 \
              --patience 80 \
              --lr "$lr" \
              --threshold_p "$threshold" \
              --epochs 400 \
              --augment none \
              --num_workers 12 \
              --seed "$seed"

            docker wait fvae

            docker rm -f fvae

            echo "=== Finished combination ==="
            echo ""
          done
        done
      done
    done
  done
done
