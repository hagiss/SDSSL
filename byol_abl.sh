#!/bin/sh
python3 pl_train.py --st_inter False --p_loss 0.0 --p2_loss 0.0
python3 pl_train.py --st_inter False --p_loss 0.0 --p2_loss 1.0
python3 pl_train.py --st_inter True --p_loss 1.0 --p2_loss 0.0
python3 pl_train.py --st_inter True --p_loss 0.0 --p2_loss 0.0