#!/bin/bash
#SBATCH --account=def-corbeilj
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=4000M               # memory (per node)
#SBATCH --time=0-08:00            # time (DD-HH:MM)
python learning_phase.py
