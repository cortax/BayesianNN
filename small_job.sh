#!/bin/bash
#SBATCH --time=11:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --account=def-pager47
#SBATCH --array=0-6
python main.py $SLURM_ARRAY_TASK_ID 35000
