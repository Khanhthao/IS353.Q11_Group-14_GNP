#!/bin/bash

encoder="google/flan-t5-small"
max_seq_len=128
seed=1

python3 -u llm.py \
  --dataset piqa \
  --encoder $encoder \
  --max_seq_len $max_seq_len \
  --seed $seed
