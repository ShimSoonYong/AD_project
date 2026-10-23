#!/bin/bash

./scripts/fmemae.sh &
./scripts/fvae.sh &

wait

echo "All model fittings running"
