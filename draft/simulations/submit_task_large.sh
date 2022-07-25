#! /bin/bash

module load slurm
module load disBatch/beta

sbatch -p ccn -n40 -c2 disBatch task_file_large.db
