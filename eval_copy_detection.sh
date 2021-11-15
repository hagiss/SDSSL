#!/bin/sh
ckpt="/data/byol-pytorch/checkpoints/vit_small/simclr_base.ckpt"
SET=$(seq 1 12)

for i in $SET
do
    echo "Running loop seq "$i
    python3 eval_copydetection.py --n $i --ckpt $ckpt --st_inter False
done