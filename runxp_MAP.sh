#!/bin/bash
#SBATCH --array=0-2%500
#SBATCH --time=4:00:00
#SBATCH --account=rrg-corbeilj-ac
#SBATCH --gres=gpu:0              # Number of GPUs (per node)
#SBATCH --mem=8G               # memory (per node)
python -m Experiments.Foong_L1W50.MAP --max_iter=100000 --max_iter=100000 --learning_rate=0.05 --min_lr=0.0005 --patience=200 --lr_decay=0.9 --gamma_alpha=1.0 --gamma_beta=1.0




