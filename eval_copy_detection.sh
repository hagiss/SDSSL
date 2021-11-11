#!/bin/sh

SET=$(seq 1 12)

for i in $SET
do
    echo "Running loop seq "$i
    python3 eval_copydetection.py --n $i
done