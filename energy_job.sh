#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --account=def-pager47
#SBATCH --array=7
python main.py $SLURM_ARRAY_TASK_ID 50000

