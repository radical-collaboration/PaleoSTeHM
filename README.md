[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/Development?labpath=binder%2FPaleoSTeHM_examples.ipynb)

# PaleoSTeHM - A modern, scalable Spatio-Temporal Hierarchical Modeling framework for paleo-environmental data
By [Rutgers Earth System Science & Policy Lab](https://earthscipol.net/)

## Project overview
This repository contains the Jupyter Notebook based tutorials for PaleoSTeHM project, which will develop a framework for spatiotemporal hierarchical modeling of paleodata that builds upon modern, scalable software infrastructure for machine learning. By leveraging an existing, widely used machine-learning framework at the base level, PaleoSTeHM will be built to take advantage of current and future computational advances without modifications to the user-facing product. It will facilitate the incorporation of complex likelihood structures, including the embedding of physical simulation models, and thus pave the way for further advances in paleo-modeling. 

PaleoSTeHM is a paleo-environmental modelling faced API based upon [pyro](https://pyro.ai/), a universal probabilistic programming language supported by [PyTorch](https://pytorch.org/) on the backend. Therefore, it not only supports probablistic programming not also the auto-differentiation and GPU accelration. This API will provide easy-to-use models to infer the spatio or spatio-temporal variation of environmental change from geological data, with tutorials
that cover the background theories and hand-on practicals. 

Please note that this repository is under active developmennt by Yucheng Lin, Alex Reed and Robert Kopp. If you have any questions please contact Yucheng: yc.lin@rutgers.edu. 

## Installation

```
git clone https://github.com/radical-collaboration/PaleoSTeHM.git
cd PaleoSTeHM/
pip install -r requirements.txt
```

PaleoSTeHM was developed under python version 3.7.3 and Jupyter Notebook 5.4.0. 

## Tutorial Contents
### Introduction
  - **[PaleoSTeHM background and data level modelling](Tutorials/1.Introduction/1.Introduction.ipynb)** - Introduction to Bayesian Hierachical modelling, which consist of data level, process level and parameter level modelling. This notebook will cover data level modelling with an illustrative example of building data level model for coral-based sea-level records from the Great Barrier Reef. 
  - **[Process level and parameter level modelling](Tutorials/1.Introduction/2.Process_level_modelling.ipynb)** - Process level and parameter level modelling tutorial, which will cover a range of process level model for paleo-environmental research, including hand-drawn model, linear model, change-point model, Gaussian Process model, physical model, and mixed physical and statistical model.
  - **[Analysis Choice](Tutorials/1.Introduction/3.Analysis_Choice.ipynb)** - Analysis choice tutorial, which will cover the difference between a deterministic model and a probablistic model (we will mostly use a Bayesian model), the optimization methods for deterministic model (least sqaures and gradient-based optimization) and inference method for Bayesian model, including Markov Chain Monte Carlo, Variational Bayes and Empirical Bayes.

### Temporal Gaussian Process
  - **[Gaussian Process background](Tutorials/2.Temporal_GP/4.GP_background.ipynb)** - Gaussian Process background tutorial, which covers the basics of Gaussian Process, including calculating squared distance matrix and covariance matrix, hyperparameter definitions and their impact, conditional probability and hyperparameter optimization.
  - **[Gaussian Process kernels and kernel operations](Tutorials/2.Temporal_GP/5.GP_kernels_and_operation.ipynb)** - Gaussian Process kernel tutorial, which introduces some popular GP kernels and instructions on how to combine different kernels. 
  - **[Incorporate temporal uncertainty in Gaussian Process](Tutorials/2.Temporal_GP/6.Temporal_uncer.ipynb)**  - This tutorial introduce two commonly-used methods for Gaussian Process: noisy-input Gaussian Process and errors-in-variable Gaussian Process. 
  - **[Holocene sea-level analysis in New Jersey and Northern North Carolina](Tutorials/2.Temporal_GP/7.NJ_NNC_RSL.ipynb)**  - This tutorial will use the contents covered above in order to replicate Gaussian Process modelling results in New Jersey and Northern North Carolina as in [Ashe et al., 2019](https://www.sciencedirect.com/science/article/abs/pii/S0277379118302130).
### Spatio-temporal Gaussian Process 
 - **[Spatio-temporal Gaussian Process Background ](Tutorials/3.ST_GP/8.ST_GP.ipynb)** - Background of building a Sptio-temporal Gaussian Process model.
 - **[Spatio-temporal Gaussian Process with multiple kernels](Tutorials/3.ST_GP/9.STGP_kernels.ipynb)** - Spatio-temporal Gaussian Process with multiple kernels with instructions on kernel decomposition. 
 - **[Spatio-temporal Gassusian Process with Physical Models](Tutorials/3.ST_GP/10.STGP_with_physical_model.ipynb)** - Spatio-temporal Gaussian Process in combination with one or multiple physical models.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details

## Acknowledgements

This work is supported by the National Science Foundation under awards 2002437 and 2148265. The opinions, findings, and conclusions or recommendations expressed are those of the authors and do not necessarily reflect the views of the National Science Foundation.
