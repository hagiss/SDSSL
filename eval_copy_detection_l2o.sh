#!/bin/sh
ckpt="/data/byol-pytorch/checkpoints/vit_small/simclr_l2o.ckpt"
SET=$(seq 6 12)

for i in $SET
do
    echo "Running loop seq "$i
    python3 eval_copydetection.py --n $i --ckpt $ckpt --st_inter True
done