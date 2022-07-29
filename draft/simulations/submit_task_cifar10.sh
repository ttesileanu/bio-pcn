#! /bin/bash

module load slurm
module load disBatch/beta

sbatch -p ccn -n40 -c2 --job-name cifar10 disBatch task_file_cifar10.db
