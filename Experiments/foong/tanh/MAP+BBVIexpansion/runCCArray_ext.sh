#!/bin/bash
#SBATCH --array=500-1000%500
#SBATCH --time=12:00:00
#SBATCH --account=rrg-corbeilj-ac
#SBATCH --gres=gpu:0              # Number of GPUs (per node)
#SBATCH --mem=4G               # memory (per node)
python learning_job_ext.py $SLURM_ARRAY_TASK_ID




