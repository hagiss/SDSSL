#!/bin/sh
python3 pl_train_moco.py --st_inter False --pred2 0.0 --const_ratio False --pred_loss 1.0 --max_epochs 100
python3 pl_train_moco.py --st_inter True --pred2 0.0 --const_ratio True --pred_loss 1.0 --max_epochs 100
python3 pl_train_moco.py --st_inter True --pred2 0.0 --const_ratio False --pred_loss 0.0 --max_epochs 100
python3 pl_train_moco.py --st_inter True --pred2 0.0 --const_ratio False --pred_loss 1.0 --max_epochs 100
python3 pl_train_moco.py --st_inter False --pred2 1.0 --const_ratio False --pred_loss 0.0 --max_epochs 100