#!/bin/bash

lr_options=("0.001" "0.0005" "0.0001")
wd_options=("1e-4" "1e-5")
criterons=("MSELoss" "L1Loss")
memory_sizes=("50" "100" "200")
latent_dims=("32" "64" "128")
batch_sizes=("64" "128")

for lr in "${lr_options[@]}"; do
  for wd in "${wd_options[@]}"; do
    for mem in "${memory_sizes[@]}"; do
      for lat in "${latent_dims[@]}"; do
        for bs in "${batch_sizes[@]}"; do
          for cr in "${criterons[@]}"; do
            echo "=== Running with criterion=$cr, lr=$lr, weight_decay=$wd, memory_size=$mem, latent_dim=$lat, batch_size=$bs ==="

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
              --threshold 0.04 \
              --train_criterion "$cr" \
              --score_criterion "$cr" \
              --optimizer AdamW \
              --compactness_weight 0.0 \
              --separateness_weight 0.0 \
              --memory_margin 0.2 \
              --entropy_weight 1.3e-6 \
              --weight_decay "$wd" \
              --latent_dim "$lat" \
              --hidden_dim 256 \
              --memory_size "$mem" \
              --temperature 1.0 \
              --shrink_threshold 0.01 \
              --patience 80 \
              --lr "$lr" \
              --epochs 400 \
              --augment none \
              --batch_size "$bs" \
              --num_workers 12 \
              --seed 42

            docker wait fvae

            echo "=== Finished combination ==="
            echo ""
          done
        done
      done
    done
  done
done
