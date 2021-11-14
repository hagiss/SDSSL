#!/bin/sh
ckpt="/data/byol-pytorch/checkpoints/vit_small/simclr_l2o.ckpt"

echo "paris"
python3 eval_image_retrieval.py --st_inter True --ckpt $ckpt --n 1
echo "oxford"
python3 eval_image_retrieval.py --st_inter True --ckpt $ckpt --n 1 --dataset roxford5k
