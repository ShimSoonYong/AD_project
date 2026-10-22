#!/bin/bash

./scripts/ae.sh &
./scripts/cae.sh &
./scripts/pcae.sh &
./scripts/pcr.sh &
./scripts/sae.sh &
./scripts/sattn.sh &
./scripts/pcsae.sh &
./scripts/pccae.sh &
./scripts/pcattn.sh &
./scripts/convsumae.sh &
./scripts/pcconvsumae.sh

wait

echo "All model fittings running"
