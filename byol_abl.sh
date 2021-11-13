#!/bin/sh
python3 pl_train.py --st_inter False --p_loss 0.0 --p2_loss 0.0
python3 pl_train.py --st_inter True --ratio 0.6 --p_loss 1.0 --p2_loss 0.0 --proj_bn True --pred_bn True
python3 pl_train.py --st_inter True --ratio 0.6 --p_loss 1.0 --p2_loss 0.0 --proj_bn True --pred_bn False
python3 pl_train.py --st_inter True --ratio 0.6 --p_loss 1.0 --p2_loss 0.0 --proj_bn False --pred_bn True
python3 pl_train.py --st_inter True --ratio 0.6 --p_loss 1.0 --p2_loss 0.0
python3 pl_train.py --st_inter False --p_loss 0.0 --p2_loss 1.0
python3 pl_train.py --st_inter True --ratio 0.6 --p_loss 0.0 --p2_loss 0.0
