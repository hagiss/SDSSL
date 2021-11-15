#!/bin/sh
ckpt="/data/byol-pytorch/checkpoints/vit_small/simclr_l2o.ckpt"

python3 transfer.py --lr 3e-4 --weight-decay 0.1 --warmup-epochs 3 --drop-path 0 --data-set pets --data-path ../data/pets --reprob 0.0 --mixup 0.8 --cutmix 0 --ckpt $ckpt --st_inter True --knn True
python3 transfer.py --lr 3e-4 --weight-decay 0.3 --warmup-epochs 3 --drop-path 0.1 --data-set flowers --data-path ../data/flowers --reprob 0.25 --mixup 0 --cutmix 0 --ckpt $ckpt --st_inter True --knn True
python3 transfer.py --lr 3e-4 --weight-decay 0.1 --warmup-epochs 3 --drop-path 0.1 --data-set cifar100 --data-path ../data/ --reprob 0.0 --mixup 0.5 --cutmix 1 --ckpt $ckpt --st_inter True --knn True
python3 transfer.py --lr 3e-4 --weight-decay 0.1 --warmup-epochs 3 --drop-path 0.1 --data-set cifar10 --data-path ../data/ --reprob 0.0 --mixup 0.8 --cutmix 1 --ckpt $ckpt --st_inter True --knn True
#
python3 transfer.py --lr 3e-4 --weight-decay 0.1 --warmup-epochs 3 --drop-path 0.1 --data-set pets --data-path ../data/pets --reprob 0.0 --mixup 0.5 --cutmix 0 --ckpt $ckpt --st_inter True --batch_size_per_gpu 128
python3 transfer.py --lr 3e-4 --weight-decay 0.3 --warmup-epochs 3 --drop-path 0.1 --data-set flowers --data-path ../data/flowers --reprob 0.0 --mixup 0 --cutmix 0 --ckpt $ckpt --st_inter True --batch_size_per_gpu 128
python3 transfer.py --lr 3e-4 --weight-decay 0.3 --warmup-epochs 3 --drop-path 0.1 --data-set cifar100 --data-path ../data/ --reprob 0.0 --mixup 0.5 --cutmix 1 --ckpt $ckpt --st_inter True --batch_size_per_gpu 128
python3 transfer.py --lr 3e-4 --weight-decay 0.1 --warmup-epochs 3 --drop-path 0.1 --data-set cifar10 --data-path ../data/ --reprob 0.0 --mixup 0.8 --cutmix 1 --ckpt $ckpt --st_inter True --batch_size_per_gpu 128