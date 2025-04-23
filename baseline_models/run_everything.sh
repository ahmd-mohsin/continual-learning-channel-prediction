#!/bin/bash
# Run transformer model on umi_compact
python main.py --ext mat --model_type TRANS 

# Run transformer model on umi_dense
python main.py --ext mat --model_type GRU 

# Run transformer model on umi_standard
python main.py --ext mat --model_type LSTM 

