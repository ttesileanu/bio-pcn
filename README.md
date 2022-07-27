Constrained predictive coding
=============================

This is the code used to generate the results from *Constrained Predictive Coding as a Biologically
Plausible Model of the Cortical Hierarchy*. It runs on Python 3.9. To install, run the following in the repo folder:

    conda env create -f conda.yml
    conda activate cpcn
    pip install .

 You can instead use

    pip install -e .

to create an editable install – useful for development.

The `submit_task_*` scripts in `draft/hyper` run the hyperoptimizations. These are written for the
Flatiron Institute's SLURM system, and use [disBatch](https://github.com/flatironinstitute/disBatch).
This is equivalent to running each line of the corresponding `task_file_*.db`.

After the hyperoptimization runs are completed, similar `submit_task_*` scripts are available in
`draft/simulations`. These generate the data needed for making the figures.

Figures are generated using the various `make_*_figures_*.py` scripts in `draft`. Some exploratory
code can be found in `sandbox`.