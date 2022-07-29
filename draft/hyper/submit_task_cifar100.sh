#! /bin/bash

module load slurm
module load disBatch/beta

sbatch -p ccn -n80 -c2 --job-name cifar100 disBatch task_file_cifar100.db
