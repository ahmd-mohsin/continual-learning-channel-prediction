#!/bin/bash



# Test only on umi_dense
python nmse.py --ext mat --model_type GRU --file_path "../dataset/outputs/umi_compact_conf_2tx_2rx." --test_file_path "../dataset/outputs/umi_dense_conf_8tx_2rx."
# Test only on umi_standard
python nmse.py --ext mat --model_type GRU --file_path "../dataset/outputs/umi_compact_conf_2tx_2rx." --test_file_path "../dataset/outputs/umi_standard_conf_16tx_2rx."
python nmse.py --ext mat --model_type GRU --file_path "../dataset/outputs/umi_compact_conf_2tx_2rx." --test_file_path "../dataset/outputs/umi_compact_conf_2tx_2rx."
# Test only on umi_compact
python nmse.py --ext mat --model_type GRU --file_path "../dataset/outputs/umi_dense_conf_8tx_2rx." --test_file_path "../dataset/outputs/umi_compact_conf_2tx_2rx."
# Test only on umi_standard
python nmse.py --ext mat --model_type GRU --file_path "../dataset/outputs/umi_dense_conf_8tx_2rx." --test_file_path "../dataset/outputs/umi_standard_conf_16tx_2rx."
python nmse.py --ext mat --model_type GRU --file_path "../dataset/outputs/umi_dense_conf_8tx_2rx." --test_file_path "../dataset/outputs/umi_dense_conf_8tx_2rx."
# Test only on umi_compact
python nmse.py --ext mat --model_type GRU --file_path "../dataset/outputs/umi_standard_conf_16tx_2rx." --test_file_path "../dataset/outputs/umi_compact_conf_2tx_2rx."
# Test only on umi_dense
python nmse.py --ext mat --model_type GRU --file_path "../dataset/outputs/umi_standard_conf_16tx_2rx." --test_file_path "../dataset/outputs/umi_dense_conf_8tx_2rx."
python nmse.py --ext mat --model_type GRU --file_path "../dataset/outputs/umi_standard_conf_16tx_2rx." --test_file_path "../dataset/outputs/umi_standard_conf_16tx_2rx."

##############################################################################
# Test only on umi_dense
python nmse.py --ext mat --model_type LSTM --file_path "../dataset/outputs/umi_compact_conf_2tx_2rx." --test_file_path "../dataset/outputs/umi_dense_conf_8tx_2rx."
# Test only on umi_standard
python nmse.py --ext mat --model_type LSTM --file_path "../dataset/outputs/umi_compact_conf_2tx_2rx." --test_file_path "../dataset/outputs/umi_standard_conf_16tx_2rx."
python nmse.py --ext mat --model_type LSTM --file_path "../dataset/outputs/umi_compact_conf_2tx_2rx." --test_file_path "../dataset/outputs/umi_compact_conf_2tx_2rx."
# Test only on umi_compact
python nmse.py --ext mat --model_type LSTM --file_path "../dataset/outputs/umi_dense_conf_8tx_2rx." --test_file_path "../dataset/outputs/umi_compact_conf_2tx_2rx."
# Test only on umi_standard
python nmse.py --ext mat --model_type LSTM --file_path "../dataset/outputs/umi_dense_conf_8tx_2rx." --test_file_path "../dataset/outputs/umi_standard_conf_16tx_2rx."
python nmse.py --ext mat --model_type LSTM --file_path "../dataset/outputs/umi_compact_conf_2tx_2rx." --test_file_path "../dataset/outputs/umi_compact_conf_2tx_2rx."
# Test only on umi_compact
python nmse.py --ext mat --model_type LSTM --file_path "../dataset/outputs/umi_standard_conf_16tx_2rx." --test_file_path "../dataset/outputs/umi_compact_conf_2tx_2rx."
# Test only on umi_dense
python nmse.py --ext mat --model_type LSTM --file_path "../dataset/outputs/umi_standard_conf_16tx_2rx." --test_file_path "../dataset/outputs/umi_dense_conf_8tx_2rx."
python nmse.py --ext mat --model_type LSTM --file_path "../dataset/outputs/umi_standard_conf_16tx_2rx." --test_file_path "../dataset/outputs/umi_standard_conf_16tx_2rx."
##############################################################################
# Test only on umi_dense
python nmse.py --ext mat --model_type TRANS --file_path "../dataset/outputs/umi_compact_conf_2tx_2rx." --test_file_path "../dataset/outputs/umi_dense_conf_8tx_2rx."
# Test only on umi_standard
python nmse.py --ext mat --model_type TRANS --file_path "../dataset/outputs/umi_compact_conf_2tx_2rx." --test_file_path "../dataset/outputs/umi_standard_conf_16tx_2rx."
python nmse.py --ext mat --model_type TRANS --file_path "../dataset/outputs/umi_compact_conf_2tx_2rx." --test_file_path "../dataset/outputs/umi_compact_conf_2tx_2rx."
# Test only on umi_compact
python nmse.py --ext mat --model_type TRANS --file_path "../dataset/outputs/umi_dense_conf_8tx_2rx." --test_file_path "../dataset/outputs/umi_compact_conf_2tx_2rx."
# Test only on umi_standard
python nmse.py --ext mat --model_type TRANS --file_path "../dataset/outputs/umi_dense_conf_8tx_2rx." --test_file_path "../dataset/outputs/umi_standard_conf_16tx_2rx."
python nmse.py --ext mat --model_type TRANS --file_path "../dataset/outputs/umi_compact_conf_2tx_2rx." --test_file_path "../dataset/outputs/umi_compact_conf_2tx_2rx."
# Test only on umi_compact
python nmse.py --ext mat --model_type TRANS --file_path "../dataset/outputs/umi_standard_conf_16tx_2rx." --test_file_path "../dataset/outputs/umi_compact_conf_2tx_2rx."
# Test only on umi_dense
python nmse.py --ext mat --model_type TRANS --file_path "../dataset/outputs/umi_standard_conf_16tx_2rx." --test_file_path "../dataset/outputs/umi_dense_conf_8tx_2rx."
python nmse.py --ext mat --model_type TRANS --file_path "../dataset/outputs/umi_standard_conf_16tx_2rx." --test_file_path "../dataset/outputs/umi_standard_conf_16tx_2rx."
