#!/bin/bash
#SBATCH --array=0-359%100
#SBATCH --time=8:00:00
#SBATCH --account=rrg-corbeilj-ac
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=4000M               # memory (per node)
python learning_job.py $SLURM_ARRAY_TASK_ID




