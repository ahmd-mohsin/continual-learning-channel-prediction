#!/bin/bash
# Test only on umi_dense
python nmse.py --ext mat --model_type GRU --file_path "../dataset/outputs/umi_compact_8Tx_2Rx." --test_file_path "../dataset/outputs/umi_dense_8Tx_2Rx."
# Test only on umi_standard
python nmse.py --ext mat --model_type GRU --file_path "../dataset/outputs/umi_compact_8Tx_2Rx." --test_file_path "../dataset/outputs/umi_standard_8Tx_2Rx."
python nmse.py --ext mat --model_type GRU --file_path "../dataset/outputs/umi_compact_8Tx_2Rx." --test_file_path "../dataset/outputs/umi_compact_8Tx_2Rx."
# Test only on umi_compact
python nmse.py --ext mat --model_type GRU --file_path "../dataset/outputs/umi_dense_8Tx_2Rx." --test_file_path "../dataset/outputs/umi_compact_8Tx_2Rx."
# Test only on umi_standard
python nmse.py --ext mat --model_type GRU --file_path "../dataset/outputs/umi_dense_8Tx_2Rx." --test_file_path "../dataset/outputs/umi_standard_8Tx_2Rx."
python nmse.py --ext mat --model_type GRU --file_path "../dataset/outputs/umi_dense_8Tx_2Rx." --test_file_path "../dataset/outputs/umi_dense_8Tx_2Rx."
# Test only on umi_compact
python nmse.py --ext mat --model_type GRU --file_path "../dataset/outputs/umi_standard_8Tx_2Rx." --test_file_path "../dataset/outputs/umi_compact_8Tx_2Rx."
# Test only on umi_dense
python nmse.py --ext mat --model_type GRU --file_path "../dataset/outputs/umi_standard_8Tx_2Rx." --test_file_path "../dataset/outputs/umi_dense_8Tx_2Rx."
python nmse.py --ext mat --model_type GRU --file_path "../dataset/outputs/umi_standard_8Tx_2Rx." --test_file_path "../dataset/outputs/umi_standard_8Tx_2Rx."

##############################################################################
# Test only on umi_dense
python nmse.py --ext mat --model_type LSTM --file_path "../dataset/outputs/umi_compact_8Tx_2Rx." --test_file_path "../dataset/outputs/umi_dense_8Tx_2Rx."
# Test only on umi_standard
python nmse.py --ext mat --model_type LSTM --file_path "../dataset/outputs/umi_compact_8Tx_2Rx." --test_file_path "../dataset/outputs/umi_standard_8Tx_2Rx."
python nmse.py --ext mat --model_type LSTM --file_path "../dataset/outputs/umi_compact_8Tx_2Rx." --test_file_path "../dataset/outputs/umi_compact_8Tx_2Rx."
# Test only on umi_compact
python nmse.py --ext mat --model_type LSTM --file_path "../dataset/outputs/umi_dense_8Tx_2Rx." --test_file_path "../dataset/outputs/umi_compact_8Tx_2Rx."
# Test only on umi_standard
python nmse.py --ext mat --model_type LSTM --file_path "../dataset/outputs/umi_dense_8Tx_2Rx." --test_file_path "../dataset/outputs/umi_standard_8Tx_2Rx."
python nmse.py --ext mat --model_type LSTM --file_path "../dataset/outputs/umi_compact_8Tx_2Rx." --test_file_path "../dataset/outputs/umi_compact_8Tx_2Rx."
# Test only on umi_compact
python nmse.py --ext mat --model_type LSTM --file_path "../dataset/outputs/umi_standard_8Tx_2Rx." --test_file_path "../dataset/outputs/umi_compact_8Tx_2Rx."
# Test only on umi_dense
python nmse.py --ext mat --model_type LSTM --file_path "../dataset/outputs/umi_standard_8Tx_2Rx." --test_file_path "../dataset/outputs/umi_dense_8Tx_2Rx."
python nmse.py --ext mat --model_type LSTM --file_path "../dataset/outputs/umi_standard_8Tx_2Rx." --test_file_path "../dataset/outputs/umi_standard_8Tx_2Rx."
##############################################################################
# Test only on umi_dense
python nmse.py --ext mat --model_type TRANS --file_path "../dataset/outputs/umi_compact_8Tx_2Rx." --test_file_path "../dataset/outputs/umi_dense_8Tx_2Rx."
# Test only on umi_standard
python nmse.py --ext mat --model_type TRANS --file_path "../dataset/outputs/umi_compact_8Tx_2Rx." --test_file_path "../dataset/outputs/umi_standard_8Tx_2Rx."
python nmse.py --ext mat --model_type TRANS --file_path "../dataset/outputs/umi_compact_8Tx_2Rx." --test_file_path "../dataset/outputs/umi_compact_8Tx_2Rx."
# Test only on umi_compact
python nmse.py --ext mat --model_type TRANS --file_path "../dataset/outputs/umi_dense_8Tx_2Rx." --test_file_path "../dataset/outputs/umi_compact_8Tx_2Rx."
# Test only on umi_standard
python nmse.py --ext mat --model_type TRANS --file_path "../dataset/outputs/umi_dense_8Tx_2Rx." --test_file_path "../dataset/outputs/umi_standard_8Tx_2Rx."
python nmse.py --ext mat --model_type TRANS --file_path "../dataset/outputs/umi_compact_8Tx_2Rx." --test_file_path "../dataset/outputs/umi_compact_8Tx_2Rx."
# Test only on umi_compact
python nmse.py --ext mat --model_type TRANS --file_path "../dataset/outputs/umi_standard_8Tx_2Rx." --test_file_path "../dataset/outputs/umi_compact_8Tx_2Rx."
# Test only on umi_dense
python nmse.py --ext mat --model_type TRANS --file_path "../dataset/outputs/umi_standard_8Tx_2Rx." --test_file_path "../dataset/outputs/umi_dense_8Tx_2Rx."
python nmse.py --ext mat --model_type TRANS --file_path "../dataset/outputs/umi_standard_8Tx_2Rx." --test_file_path "../dataset/outputs/umi_standard_8Tx_2Rx."
