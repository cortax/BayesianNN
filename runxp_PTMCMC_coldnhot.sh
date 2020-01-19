#!/bin/bash
#SBATCH --array=0-2%10
#SBATCH --time=12:00:00
#SBATCH --account=rrg-corbeilj-ac
#SBATCH --gres=gpu:0              # Number of GPUs (per node)
#SBATCH --mem=32G               # memory (per node)
python PTMCMC_coldnhot.py --layerwidth 50 --nblayers 1 --tag $SLURM_ARRAY_TASK_ID --nbchunk 10 --chunksize 100000 --device cpu 



