#! /bin/bash

module load slurm
module load disBatch

sbatch -p ccn --constraint=skylake -n 80 --ntasks-per-node 40 disBatch task_file_rho_dep_mmill_large.db
