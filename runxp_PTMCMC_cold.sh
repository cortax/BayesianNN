#!/bin/bash
#SBATCH --array=0-10%10
#SBATCH --time=12:00:00
#SBATCH --account=rrg-corbeilj-ac
#SBATCH --gres=gpu:0              # Number of GPUs (per node)
#SBATCH --mem=32G               # memory (per node)
python PTMCMC_cold.py $SLURM_ARRAY_TASK_ID 10 100000




