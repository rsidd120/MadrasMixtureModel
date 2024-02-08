# MadrasMixtureModel
Clustering heterogeneous tabular data, and generating synthetic tabular data

This is code accompanying Chandrani Kumari and Rahul Siddharthan, “MMM and MMMSynth: Clustering of heterogeneous tabular data, and synthetic data generation” [preprint](https://arxiv.org/abs/2310.19454).

The code is tested in Julia 1.10.0.  This repository will be updated with more usable versions.  Currently it has

* `mmm.jl` -- a program for clustering heterogeneous data. Run with `-h` for options. Outputs a “labels” file giving the cluster label of each row.
* `mmm_lib.jl` -- functions that are imported into `mmm.jl`, for now save it in the same directory as `mmm.jl`. A future version will convert it into a proper module
* `synth_data.jl` -- a program to generate synthetic tabular datasets consisting of categorical and numeric columns, used for benchmarks in the manuscript. Run with `-h` for help
* `GenerateSyntheticData.jl` -- functions to use a clustering of a real tabular dataset to generate a synthetic dataset. This constitutes the MMMSynth algorithm but as of now it is not a runnable program, but can be called from other programs or a Jupyter session.
* `MMM_demo.ipynb` -- a sample Jupyter notebook showing how to run MMM from inside a Jupyter notebook
