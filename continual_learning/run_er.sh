#!/bin/bash

# Define arrays for sampling techniques and model types
sampling_techniques=("reservoir" "lars")
model_types=("LSTM" "GRU" "TRANS")

# Loop over all combinations
for sampling in "${sampling_techniques[@]}"
do
  for model in "${model_types[@]}"
  do
    echo "Running model: $model with sampling: $sampling"
    python er_updated.py --model_type "$model" --sampling "$sampling"
  done
done
