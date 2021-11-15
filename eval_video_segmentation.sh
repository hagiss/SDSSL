#!/bin/sh
ckpt="/data/byol-pytorch/checkpoints/vit_small/byol_base.ckpt"
output_dir="/data/video_segmentation/byol_base_"

SET=$(seq 1 12)
for i in $SET
do
    echo "Running loop seq "$i
    # some instructions
    python3 eval_video_segmentation.py --st_inter False --ckpt $ckpt --output_dir "${output_dir}${i}" --n "$i"
done