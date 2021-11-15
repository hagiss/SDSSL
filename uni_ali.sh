#!/bin/sh
ckpt_base=""
ckpt_l2o=""
python3 uni_ali.py --ckpt_base $ckpt_base --ckpt_l2o $ckpt_l2o --normalize True
python3 uni_ali.py --ckpt_base $ckpt_base --ckpt_l2o $ckpt_l2o --normalize False