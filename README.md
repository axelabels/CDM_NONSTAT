# A New Approach to Handle Non-stationarity in Collective Decision-Making with Experts

This repository contains the code to reproduce the results of our paper "A New Approach to Handle Non-stationarity in Collective Decision-Making with Experts". 

Use `runner.py` to generate the result files:
> python runner.py seed 

Plots can subsequently be generated through the `plots.ipynb` notebook.

If you'd like to plot Figure 7 (normalized model error against average reward), pass the --extra-figure flag to runner.py. This will significantly slow down the experiment.
> python runner.py seed --extra-figure