#! /bin/bash

module load slurm
module load disBatch/beta

sbatch -p ccn -n40 -c2 --job-name lfw disBatch task_file_lfw.db
