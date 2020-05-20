#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --ntasks=7
#SBATCH --cpus-per-task=1 
#SBATCH --mem-per-cpu=32G
#SBATCH --account=rrg-corbeilj-ac
#SBATCH --array=0-6
python main.py $SLURM_ARRAY_TASK_ID 35000
