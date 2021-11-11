#!/bin/sh
python3 pl_train_moco.py --st_inter False --pred2 0.0 --const_ratio False --pred_loss 1.0
python3 pl_train_moco.py --st_inter True --pred2 0.0 --const_ratio False --pred_loss 1.0