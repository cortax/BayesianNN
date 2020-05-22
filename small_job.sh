#!/bin/bash
#SBATCH --time=13:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --account=def-pager47
#SBATCH --array=1,3
python main.py $SLURM_ARRAY_TASK_ID 40000
