#!/bin/bash
#SBATCH --array=0-359%360
#SBATCH --time=1:00:00
#SBATCH --account=rrg-corbeilj-ac
#SBATCH --gres=gpu:0              # Number of GPUs (per node)
#SBATCH --mem=32G               # memory (per node)
python evaluation_job.py $SLURM_ARRAY_TASK_ID




