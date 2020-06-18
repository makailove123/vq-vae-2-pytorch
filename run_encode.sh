#!/bin/bash

CKPT_PATH="output_full_base_cycle/ckpt/checkpoint-60000"
python encode_vqvae.py \
  --model_path "$CKPT_PATH/pytorch_model.bin" \
  --config_path "$CKPT_PATH/config.json" \
  --img_size 224 \
  --device "cuda" \
  --img_root_path "/mnt2/makai/imgs" \
  --img_key_path "/mnt2/makai/test_img_keys_2_split" \
  --output_path "/mnt2/makai/img_encode_vqvae_2" \
  --num_workers 2