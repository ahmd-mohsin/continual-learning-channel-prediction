
# Define arrays for strategies and model types
strategies=( "lwf")
model_types=("LSTM" "TRANS" "GRU" )

# Loop over all combinations
for strategy in "${strategies[@]}"; do
  for model in "${model_types[@]}"; do
    echo "Running strategy: $strategy | model: $model"
    python lwf.py --strategy "$strategy" --model_type "$model"
  done
done
