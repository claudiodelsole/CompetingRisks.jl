# Bayesian Nonparametric Competing Risks

This repository contains the source code and scripts to replicate results in the paper *Principled Estimation and Prediction with Competing Risks: A Bayesian Nonparametric Approach* by Claudio Del Sole, Antonio Lijoi and Igor Pr\"unster. 
The paper introduces a dependent nonparametric prior on the transition probabilities of a multi-state model for competing risks; transition rates are specified as kernel mixtures with respect to hierarchical completely random measures.

The core source code is organized in the form of [Julia](https://julialang.org/) package, and is available in the `/src` folder. The code implements the posterior sampling algorithms for mixture hazard models with hierarchical generalized gamma completely random measures, as detailed in Section S3 (Supplementary Material). An implementation of similar algorithms for mixture hazard models with independent generalized gamma completely random measures is also available.

The folder `/paper_code` collects the scripts to reproduce the analyses of synthetic and clinical datasets presented in the paper (numbers, figures and tables).
Auxiliary functions to generate synthetic datasets, compute estimation errors and manipulate outputs are gathered in the `/aux_code` folder.

## Installation and usage in Julia

This repository is in the form of Julia package, created using Julia version 1.12.1 (2025-10-17). To install and load the package:
* clone this repository in a local folder;
* open a Julia REPL (i.e. the interactive command-line interface for Julia) inside that folder;
* create a Julia environment by running `using Pkg` and `Pkg.activate(".")`: the local environment is named `CompetingRisks` and contains the information in `Project.toml`;
* download and install the required dependencies by running `Pkg.instantiate()`: the exact version of each dependency is listed in `Manifest.toml`;
* load the package by running `using CompetingRisks`.

The package exports data structures and function for model specification, posterior sampling and estimation, and plotting.
The structure `CompetingRisksModel` contains the hyperparameters of the mixing random measures and the kernel specification. The available kernel types are `DykstraLaudKernel`, `OrnsteinUhlenbeckKernel` and `RectangularKernel`; see Section 2 for details.
The main function for posterior inference is `posterior_sampling`; the output includes data structures collecting posterior samples and diagnostics, namely `Estimator` and `Parameters`.
Posterior estimates and pointwise credible intervals of functionals of interest are obtained using the functions `estimate_survival` for the survival function, `estimate_incidence` for the cause-specific incidence and subdistribution functions, and `estimate_proportions` for prediction curves.
Functions for plotting posterior estimates, traceplots, and histograms are also available.
Further details on the arguments and outputs of exported functions can be found in their documentation, which can be accessed by running `?(function_name)`, for example `?posterior_sampling`. Likewise, details on data structures and their fields can be obtained by accessing their documentation.

A complete working example is available, for example, in the script `paper_code/illustration.jl`.

## Reproducibility workflow

The folder `/paper_code` collects the Julia scripts to perform the analyses of synthetic and clinical datasets presented in the paper, and reproduce any numbers, figures and tables:
* `illustration.jl` reproduces the synthetic dataset and analyses for the illustration in Section 7.1, including Figures 2 and 3, and Section S5.1, including Figures S1 to S6;
* `simulation_study.jl` reproduces numbers in Table 1 (Section 7.2), comparing mixture hazard models with dependent or independent mixing measures, and Figure S8; 
* `ebmt.jl` reproduces the analyses of the EBMT dataset in Sections 7.3 and S5.4, including Figures 4 and S9, and the respective comparison in Section S7.2 (Figure S13);
* `melanoma.jl` reproduces the analyses of the melanoma dataset in Section S6, including Figures 1, S10 and S11, and the respective comparison in Section S7.2 (Figure S14);
* `consistency.jl` reproduces the simulation study in Section S5.2 (Figure S7), supporting consistency of posterior estimates for the survival function;
* `comparison.jl` reproduces the synthetic dataset and analyses for the comparison in Section S7.1, including Table S1 and Figure S12.

Numerical outputs from the scripts are conveniently collected in the `\output` folder. Figures for the main manuscript are saved in the `\figures` folder; figures for the Supplementary material are saved in the `\figures_supp` folder.

The analyses in the scripts `illustration.jl`, `ebmt.jl`, `melanoma.jl` and `comparison.jl` run in few minutes on a standard desktop machine. The simulation studies in the script files `simulation_study.jl` and `consistency.jl` may need up to an hour each.

## Clinical datasets

This paper illustrates the proposed methodology on two publicly available dataset. 
* The EBMT dataset is extracted from the bone marrow transplant registry of the European Blood and Marrow Transplant (EBMT) Group, and contains 400 observations. The dataset is available in the R package [`crrSC`](https://cran.r-project.org/package=crrSC) by running `data(center)`. A description of the dataset for the purposes of this paper is in Section 7.3.
* The melanoma dataset was collected by Drzewiecki et al. (1980) at the Odense University Hospital, Denmark, on patients diagnosed with melanoma, and contains 205 observations. The dataset is available in the R package [`timereg`](https://cran.r-project.org/package=timereg) by running `data(melanoma)`. A description of the dataset is in Section S6.
For convenience, the `\data` folder contains the datasets files in plain text format, and their respective data dictionaries.
