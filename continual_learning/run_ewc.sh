#!/bin/bash

# Define arrays for sampling techniques and model types
model_types=("LSTM" "GRU" "TRANS")

# Loop over all combinations
for model in "${model_types[@]}"
do
  echo "Running model: $model"
  python ewc.py --model_type "$model"
done


bus is trah kuch likha tha keh mai apnay paper mai kya ker raha hun aur mainay EWC implement krna hai but I need trick and different methods which I can use to update and make my code better, but deep search ko thora detail mai batayein keh code bhi chahiye hoga snippets of code of different technqiues which can make my EWC better aur GPT ko EWC ka code bhi dein