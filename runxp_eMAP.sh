#!/bin/bash
#SBATCH --array=0-9%500
#SBATCH --time=6:00:00
#SBATCH --account=rrg-corbeilj-ac
#SBATCH --gres=gpu:0              # Number of GPUs (per node)
#SBATCH --mem=8G               # memory (per node)
python -m Experiments.Foong_L1W50.eMAP --ensemble_size=5 --max_iter=50000 --learning_rate=0.1 --min_lr=0.001 --patience=500 --lr_decay=0.7 --gamma_alpha=1.0 --gamma_beta=10.0




