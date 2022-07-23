#! /bin/bash

module load slurm
module load disBatch

sbatch -p ccn -n80 -c2 disBatch task_file_small.db
