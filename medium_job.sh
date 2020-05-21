#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=128G
#SBATCH --account=rrg-corbeilj-ac
#SBATCH --array=5-6
python main.py $SLURM_ARRAY_TASK_ID 50000
