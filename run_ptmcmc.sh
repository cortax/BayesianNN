#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --account=rrg-corbeilj-ac
#SBATCH --cpus-per-task=1             # Number of CPUs (per task)
#SBATCH --mem=32G               # memory (per node)
python run_ptmcmc.py $SLURM_ARRAY_TASK_ID