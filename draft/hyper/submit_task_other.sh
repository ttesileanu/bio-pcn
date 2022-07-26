#! /bin/bash

module load slurm
module load disBatch/beta

sbatch -p ccn -n80 -c2 disBatch task_file_other.db
