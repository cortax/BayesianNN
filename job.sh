#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=128G
#SBATCH --account=def-pager47
#SBATCH --array=0-6
python main.py $SLURM_ARRAY_TASK_ID 2500
