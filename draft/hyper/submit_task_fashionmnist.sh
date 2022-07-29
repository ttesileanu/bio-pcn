#! /bin/bash

module load slurm
module load disBatch/beta

sbatch -p ccn -n80 -c2 --job-name fashionmnist disBatch task_file_fashionmnist.db
