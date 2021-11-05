#!/bin/sh
python3 linear_simclr.py --lr 0.002 --st_inter False --arch vit_base --patch_size 16 --out_dim 256 --weight_decay 0.1 --weight_decay_end 0.1 --clip_grad 0 --batch_size_per_gpu 512 --epochs 100 --max_epochs 100 --mlp_hidden 4096 --warmup_epochs 40 --min_lr 0 --num_workers 4 --momentum_teacher 0.996 --dataset imagenet --name moco/vit_base_16_l2o --accelerator ddp --val_interval 10 --div 1 --ratio 0 --up 0 --t_inter False