Constrained predictive coding
=============================

This is the code used to generate the results from *Constrained Predictive Coding as a Biologically
Plausible Model of the Cortical Hierarchy*. It runs on Python 3.9. To install, first use `conda` to create an environment based on `conda.yml`. Then run `pip install .` in the main folder.

The `submit_task_*` scripts in `draft/hyper` run the hyperoptimizations, using [disBatch](https://github.com/flatironinstitute/disBatch). This is equivalent to running each line of the corresponding `task_file_*.db`. After the hyperoptimization runs are completed, similar `submit_task_*` scripts are available in `draft/simulations` that generate the runs needed for making the figures. The `run_multi_arch.sh` script also needs to be run. Figures are then generated using the various `make_*_figures_*.py` scripts in `draft`. Some exploratory code can be found in `sandbox`.