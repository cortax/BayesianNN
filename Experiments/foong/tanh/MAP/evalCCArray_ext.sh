#!/bin/bash
#SBATCH --array=1000-1439%100
#SBATCH --time=1:00:00
#SBATCH --account=rrg-corbeilj-ac
#SBATCH --gres=gpu:0              # Number of GPUs (per node)
#SBATCH --mem=32G               # memory (per node)
python evaluation_job_ext.py $SLURM_ARRAY_TASK_ID




