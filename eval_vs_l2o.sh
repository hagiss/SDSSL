#!/bin/sh
#output_dir="/data/video_segmentation/byol_base_"
output_dir_l2o="/data/video_segmentation/byol_l2o_"

SET=$(seq 1 12)
#echo "base"
#for i in $SET
#do
#    echo "Running loop seq "$i
#    # some instructions
#    python3 /data/dataset/davis2017-evaluation/evaluation_method.py --task semi-supervised --results_path "$output_dir$i" --davis_path /data/dataset/davis-2017/DAVIS/
#done

echo "l2o"
for i in $SET
do
    echo "Running loop seq "$i
    # some instructions
    python3 /data/dataset/davis2017-evaluation/evaluation_method.py --task semi-supervised --results_path "$output_dir_l2o$i" --davis_path /data/dataset/davis-2017/DAVIS/
done