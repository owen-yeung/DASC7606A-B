#!/bin/bash
# Bash script to run the Python training script

python main.py \
  --dataset cifar100 \
  --data_dir data \
  --output_dir results \
  --device cuda \
  --batch_size 128 \
  --num_epochs 300 \
  --lr 0.001 \
  --weight_decay 5e-4 \
  --early_stopping_patience 300 \
  --num_workers 4
