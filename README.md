**This repository contains the full codebase used in the manuscript:
"Population-informed Joint Calibration of Vital Rates in IPMs via Gaussian Processes and ABC"**

It includes two components:

- A simulation study under known data-generating models
- A real ecological case study using C. flava data

The methods use Gaussian Processes (GP) for flexible vital rate modelling within Integral Projection Models (IPMs), and an Approximate Bayesian Computation (ABC) method to reweight vital rate combinations using population-level information.

The recommended execution order is:  
 > **MCMC model fitting → Summary statistic selection → ABC sampling**, 

as downstream ABC components rely on MCMC output and selected statistics.

> **Note:** The scripts build their file paths relative to the working directory, so run each one from inside its own study folder (`simulation_case_study/` or `real_case_study/ABC model fitting/`).

The Gaussian process code relies on GPflow, together with TensorFlow and TensorFlow-Probability; the GLM baselines use PyMC and Bambi; and parallel computation for the ABC sampler is implemented using Ray.

## Demo

For a minimal end-to-end example on a small synthetic dataset, open [`toy_example/vignette.ipynb`](toy_example/vignette.ipynb). It runs in about a minute and needs none of the large cached files. The notebook walks through each step of this study in turn, showing the actual code and pointing to the files in both case studies that implement it.

## Precomputed results

Reproducing the published results requires the outputs of the expensive fitting steps. These are several gigabytes, so they are not kept in the repository; they are attached to a release on the [GitHub Releases page](https://github.com/zhixiaozhu/ipm-gp-abc-2025/releases) and consist of:

- `true_popu/` in both studies – the fitted MCMC samples, the fitted models, and the stored population predictions
- `ABC_details/` in both studies – the accepted ABC particles and their weights
- `IPM_matrix/` and `Marginal/` in the simulation study – the stored evaluation results
- `ss_selection/mcmc_samples/` in the real case study – the MCMC output used for summary statistic selection

To download them and place them in the correct folders, run:

```
python download_full_results.py
```

Add `--all` to also fetch the ABC particle summaries, which are large and are only needed to reproduce the efficiency figures in `r7_efficiency.ipynb` and `s5_efficiency.ipynb`.

**Files for GP posterior caches are handled differently, and are not included in the download.** ABC assembles an IPM for each of many thousands of particles, and every assembly needs each vital rate's GP posterior. Recomputing these each time would be far too slow, so they are computed once and stored in `true_popu/mcmc/Lm/`: one cached GP posterior for each MCMC sample of every vital rate. Precomputing these caches substantially reduces the cost of repeated GP prediction in ABC. Because caches are stored for every MCMC draws and vital rates, the complete cache file exceeds 50 GB. It is therefore neither kept in the repository nor included in the download. It can be regenerated locally from the MCMC samples by:

- `s1_cache_produce.py` – in the simulation study
- `r2_cache_producing_and_testing.py` – in the real case study

Run the relevant one before the ABC sampling and prediction scripts. Reproducing the figures from the downloaded results does not require it.

> **Note:** If you **only** want to try the method, or to read through how the code works, this download is not needed. Instead, use [`toy_example/vignette.ipynb`](toy_example/vignette.ipynb) for a simple demonstration.


---

## Simulation study

This section implements the full simulation workflow used to evaluate the proposed ABC_GP IPM framework. Two simulated datasets were generated under different data-generating processes:

- **`D_{glm}`**: Simulated from GLM-based vital rate models  
- **`D_{gp}`**: Simulated from GP-based vital rate models

Files with a trailing `2` in their name refer to the `D_{gp}` scenario, while those **without** it refer to `D_{glm}`. The two pipelines are otherwise structurally identical.

### Core model and Infrastructure

- `s0_class_ABCPMC.py` – ABC sampler class  
- `s0_class_GP_IPM.py` – GP-based IPM model implementation  
- `s0_fun_base.py` – Base functions used across modules  
- `s0_fun_IBMs.py` – Individual-based model (IBM) simulation functions  
- `s0_fun_IPMs.py` – Deterministic IPM operators and projections
- `s0_fun_ss.py` – Candidate summary statistics and the distance metrics used to compare simulated and observed populations

### Data generation and MCMC setup

- `s0_preprocess.ipynb` – Prepares vital rate models, simulates the datasets `D_{glm}` and `D_{gp}`, and fits GLM via MCMC  
- Folder `s0_gp_MCMC/` – Contains MCMC fitting scripts for each vital rate using GP priors, one script per vital rate and a `2`-suffixed twin of each for `D_{gp}`

### Summary Statistic Design and Evaluation

- `s1_ss_selection/` – Proposed summary statistic selection pipeline, one notebook per vital rate and a `2`-suffixed twin of each for `D_{gp}`  
- `s1_cache_produce.py`, `s1_cache_produce2.py` – Cache MCMC samples to speed up ABC sampling
- `s1_pertur_test2.ipynb` – Generates perturbation sensitivity plots (Appendix C.2); this check was only run for `D_{gp}`

### ABC sampling and Evaluation

- `s2_ABCSMC_sep.py`, `s2_ABCSMC_sep2.py` – ABC sampling runs  
- `s2_IPM_compare.ipynb`, `s2_IPM_compare2.ipynb` –  Performance comparisons  
- `s3_marginal.ipynb`, `s3_marginal2.ipynb` – Marginal posterior exploration  
- `s4_glm_recheck.ipynb` – Re-fits and re-checks the GLM on a much larger simulated population, grown by iterating the IBM for 70 steps, to separate small-sample effects from model misspecification  
- `s5_efficiency.ipynb`, `s5_efficiency2.ipynb` – Efficiency evaluation of ABC samplers

- `ABC_details/` – Stores results from ABC runs, including accepted sample indices and metadata  
  > **Note:** These files only record **indices of accepted MCMC samples**; to recover full parameter values or predictions, they must be used in conjunction with the stored MCMC samples.
  
---

## Real case study

This section applies the method to real-world demographic data from a C. Flava population.

### Directory: `real_case_study/ss_selection/`

This folder implements the full workflow for summary statistic selection and evaluation for different vital rates in the real case study. Each vital rate (e.g., survival, growth, fecundity, flowering) is treated independently following the same modular structure:

- `GP_model_fitting.py`, `GP_IPM_class.py` – Shared GP and IPM class definitions
- `s0_fun_teststats.py` – Defines candidate summary statistics used in ABC
- `s0_fun_perturbation.py`, `s0_fun_IBM.py`, `s0_initial.py` – Helper functions for IBM simulation and perturbation-based sensitivity checks
- `s1_simu_setup.py` – Sets up simulation framework for each vital rate
- `s2_simu_mcmc_*.py` – Performs MCMC fitting of GP models separately for survival, growth, fecundity, and flowering
- `s3_explore_*.ipynb` – Proposed workflow for summary statistic selection


### Directory: `real_case_study/ABC model fitting/`

- **Model and simulation functions**  
  - `r0_function_all.py`, `r0_function_seperate.py`, `r0_function_whole.py`, `r0_function.py` – Core function sets, imported as a chain in this order. `r0_function_all.py` holds the definitions, `r0_function_seperate.py` adds the machinery for fitting years separately, `r0_function_whole.py` adds the whole-period sampler, and `r0_function.py` is just a one-line shim  
  - `r0_class.py` – ABC and GP class definitions  
  - `test_stats.py` – Candidate summary statistics and distance metrics for the real data, imported by `r0_function_all.py`  
  - `r1_data_generating.py` – Preprocessing of C. flava data  
  - `r2_cache_producing_and_testing.py` – Cache pre-computation  
  - `r3_glm.py` – GLM fitting and baseline projections

- **ABC samplers**  
  - `r4_ABCSMC.py` – ABC sampler  
  - `ABC_details/` – Stores results from ABC runs, including accepted sample indices and metadata  
  > **Note:** These files only record **indices of accepted MCMC samples**; to recover full parameter values or predictions, they must be used in conjunction with the stored MCMC samples.
  
  - `r5_lambda_glm.ipynb` – GLM-based predictions
  - `r5_lambda_abc.py`, `r5_lambda_abc_z.py` – Use ABC-based posterior (`ABC_GP`)  
    - `r5_lambda_abc.py`: long-term population forecasting
    - `r5_lambda_abc_z.py`: one-step population forecasting
    
  - `r5_lambda_random.py`, `r5_lambda_random_z.py` – Use standard MCMC posterior from GP models (referred to as `GP` in the manuscript)  
    - `r5_lambda_random.py`: long-term population forecasting 
    - `r5_lambda_random_z.py`: one-step population forecasting



- **Evaluation and diagnostics**  
  - `r6_ acceptance_compare.ipynb` – Performance comparisons (note the space in the file name)  
  - `r6_ acceptance_compare_testyears.ipynb` – Long-term forecasting comparison restricted to the testing years (2009-2012), starting from the observed 2008 population structure (note the space in the file name)  
  - `r7_efficiency.ipynb` – Efficiency evaluation of ABC samplers


---

## License

This repository is distributed under the MIT License. See `LICENSE.txt` for details.




