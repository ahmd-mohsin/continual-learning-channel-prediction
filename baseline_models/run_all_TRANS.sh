#!/bin/bash

# Run transformer model on umi_compact
python main.py --ext mat --model_type TRANS --file_path "../dataset/outputs/umi_compact_conf_2tx_2rx."

# Run transformer model on umi_dense
python main.py --ext mat --model_type TRANS --file_path "../dataset/outputs/umi_dense_conf_8tx_2rx."

# Run transformer model on umi_standard
python main.py --ext mat --model_type TRANS --file_path "../dataset/outputs/umi_standard_conf_16tx_2rx."



# Test only on umi_dense
python main.py --ext mat --model_type TRANS --file_path "../dataset/outputs/umi_compact_conf_2tx_2rx." --test_only --test_file_path "../dataset/outputs/umi_dense_conf_8tx_2rx."

# Test only on umi_standard
python main.py --ext mat --model_type TRANS --file_path "../dataset/outputs/umi_compact_conf_2tx_2rx." --test_only --test_file_path "../dataset/outputs/umi_standard_conf_16tx_2rx."


# Test only on umi_compact
python main.py --ext mat --model_type TRANS --file_path "../dataset/outputs/umi_dense_conf_8tx_2rx." --test_only --test_file_path "../dataset/outputs/umi_compact_conf_2tx_2rx."

# Test only on umi_standard
python main.py --ext mat --model_type TRANS --file_path "../dataset/outputs/umi_dense_conf_8tx_2rx." --test_only --test_file_path "../dataset/outputs/umi_standard_conf_16tx_2rx."



# Test only on umi_compact
python main.py --ext mat --model_type TRANS --file_path "../dataset/outputs/umi_standard_conf_16tx_2rx." --test_only --test_file_path "../dataset/outputs/umi_compact_conf_2tx_2rx."

# Test only on umi_dense
python main.py --ext mat --model_type TRANS --file_path "../dataset/outputs/umi_standard_conf_16tx_2rx." --test_only --test_file_path "../dataset/outputs/umi_dense_conf_8tx_2rx."
