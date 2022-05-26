#! /bin/bash

module load slurm
module load disBatch

sbatch -p ccn --constraint=skylake -n 112 --ntasks-per-node 40 disBatch task_file_rho_dep.db
