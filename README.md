# Collective Intelligence in Decision-Making with Non-Stationary Experts

This repository contains the code to reproduce the results of our paper "Collective Intelligence in Decision-Making with Non-Stationary Experts". 

Use `runner.py` to generate the result files:
> python runner.py seed 

Plots can subsequently be generated through the `plots.ipynb` notebook.

If you'd like to plot Figure 7 (normalized model error against average reward), pass the --extra-figure flag to runner.py. This will significantly slow down the experiment.
> python runner.py seed --extra-figure