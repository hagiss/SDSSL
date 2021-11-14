#!/bin/sh
ckpt="/data/byol-pytorch/checkpoints/vit_small/simclr_base.ckpt"

echo "paris"
python3 eval_image_retrieval.py --st_inter False --ckpt $ckpt --n 1
echo "oxford"
python3 eval_image_retrieval.py --st_inter False --ckpt $ckpt --n 1 --dataset roxford5k
