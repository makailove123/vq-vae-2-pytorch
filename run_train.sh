#!/bin/bash

OUTPUT_ROOT="output_base"
python train_vqvae.py \
  --size 224 \
  --epochs 100 \
  --lr 3e-4 \
  --logging_steps 100 \
  --save_steps 1000 \
  --eval_steps 1000 \
  --batch_size 512 \
  --eval_path $OUTPUT_ROOT/eval \
  --config_path configs/config_base.json \
  --output_path $OUTPUT_ROOT/ckpt \
  /tmp_mnt/makai/imgs
