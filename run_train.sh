#!/bin/bash

OUTPUT_ROOT="output_full_base_cycle"
python train_vqvae.py \
  --dist_url "tcp://127.0.0.1:18881" \
  --size 224 \
  --epochs 100 \
  --lr 3e-4 \
  --sched "cycle" \
  --logging_steps 100 \
  --save_steps 1000 \
  --eval_steps 1000 \
  --batch_size 512 \
  --eval_path $OUTPUT_ROOT/eval \
  --config_path configs/config_base.json \
  --output_path $OUTPUT_ROOT/ckpt \
  --img_keys_path "/mnt2/makai/img_set_from_kept_nids_split" \
  --img_root_path "/mnt2/makai/imgs"
