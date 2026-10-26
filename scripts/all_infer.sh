#!/bin/bash

names=("ae" "convsumae" "fmemae" "fvae" "pcattn" "pcconvsumae" "pcsae" "sattn" "cae" "pcae" "pccae" "pcr" "sae")
models=("AE" "ConvSumAE" "FeatureMemAE" "FeatureVAE" "PCAttn" "PCConvSumAE" "PCSAE" "SAttn" "CAE" "PCAE" "PCCAE" "PCR" "SAE")

for ((i = 0; i < ${#names[@]}; i++)); do
  docker run \
    --name "${names[i]}" \
    --rm \
    --gpus all \
    --ipc host \
    --ulimit memlock=-1 \
    --shm-size 32gb \
    --volume $(pwd):/workspace \
    ad_project \
    python src/main.py \
    --models "${models[i]}" \
    --mode inference

  docker wait "${names[i]}"

  docker rm -f "${names[i]}"
done
