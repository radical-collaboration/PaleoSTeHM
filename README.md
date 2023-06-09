# PaleoSTeHM - A modern, scalable Spatio-Temporal Hierarchical Modeling framework for paleo-environmental data

## Overview
This repository contains the Jupyter Notebook based tutorials for PaleoSTeHM project, which will develop a framework for spatiotemporal hierarchical modeling of paleodata that builds upon modern, scalable software infrastructure for machine learning. By leveraging an existing, widely used machine-learning framework at the base level, PaleoSTeHM will be built to take advantage of current and future computational advances without modifications to the user-facing product. It will facilitate the incorporation of complex likelihood structures, including the embedding of physical simulation models, and thus pave the way for further advances in paleo-modeling.

Please note that this repository is under active developmennt by Yucheng Lin, Alex Reed and Robert Kopp. If you have any questions please contact Yucheng: yc.lin@rutgers.edu. 

## Installation

```
git clone https://github.com/radical-collaboration/PaleoSTeHM.git
cd PaleoSTeHM/
pip install -r requirements.txt
```

PaleoSTeHM was developed under python version 3.7.3 and Jupyter Notebook 5.4.0. 

## Tutorial Contents
1. Introduction
  - **[PaleoSTeHM background and data level modelling](Tutorials/1.Introduction/1.Introduction.ipynb)** - Introduction to Bayesian Hierachical modelling, which consist of data level, process level and parameter level modelling. This notebook will cover data level modelling with an illustrative example of building data level model for coral-based sea-level records from the Great Barrier Reef. 
  - **[Process level and parameter level modelling](Tutorials/1.Introduction/2.Process_level_modelling.ipynb)** - Process level and parameter level modelling tutorial, which will cover a range of process level model for paleo-environmental research, including hand-drawn model, linear model, change-point model, Gaussian Process model, physical model, mixed physical and statistical model.
  - **[Analysis Choice](Tutorials/1.Introduction/3.Analysis_Choice.ipynb)** - Analysis choice tutorial, which will cover the difference between a deterministic model and a probablistic model (we will mostly use a Bayesian model), the optimization methods for deterministic model (least sqaures and gradient-based optimization) and inference method for Bayesian model, including Markov Chain Monte Carlo, Variational Bayes and Empirical Bayes methods.
2. Temporal Gaussian Process
  - **[Gaussian Process background](Tutorials/2.Temporal_GP/4.GP_background.ipynb)** - Gaussian Process background tutorial, which covers the basics of Gaussian Process, including calculating squared distance matrix and covariance matrix, hyperparameter definitions and their impact, conditional probability and hyperparameter optimization.
  - **[Gaussian Process kernels and kernel operations](Tutorials/2.Temporal_GP/5.GP_kernels_and_operation.ipynb)** - Gaussian Process kernel tutorial, which introduces some popular GP kernels and instructions on how to combine different kernels. 
  - **[Incorporate temporal uncertainty in Gaussian Process](Tutorials/2.Temporal_GP/6.Temporal_uncer.ipynb)**  - This tutorial introduce two commonly-used methods for Gaussian Process: noisy-input Gaussian Process and errors-in-variable Gaussian Process. 
  - **[Bring it together, Holocene sea-level analysis in New Jersey and Northern North Carolina](Tutorials/2.Temporal_GP/7.NJ_NNC_RSL.ipynb)**  - This tutorial will use the contents covered above in order to replicate Gaussian Process modelling results in New Jersey and Northern North Carolina as in [Ashe et al., 2019](https://www.sciencedirect.com/science/article/abs/pii/S0277379118302130).
3. Spatio-temporal Gaussian Process (under development)

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details

## Acknowledgements

This work is supported by the National Science Foundation under awards 2002437 and 2148265. The opinions, findings, and conclusions or recommendations expressed are those of the authors and do not necessarily reflect the views of the National Science Foundation.
