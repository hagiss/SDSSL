#!/bin/sh
ckpt="/data/byol-pytorch/checkpoints/vit_small/simclr_l2o.ckpt"
output_dir="/data/video_segmentation/simclr_l2o_"

SET=$(seq 1 12)
for i in $SET
do
    echo "Running loop seq "$i
    # some instructions
    python3 eval_video_segmentation.sh --st_inter True --ckpt $ckpt --output_dir output_dir$i
done