# PaleoSTeHM - A modern, scalable Spatio-Temporal Hierarchical Modeling framework for paleo-environmental data

## Overview
Test!
This repository contains the Jupyter Notebook based tutorials for PaleoSTeHM project, which will develop a framework for spatiotemporal hierarchical modeling of paleodata that builds upon modern, scalable software infrastructure for machine learning. By leveraging an existing, widely used machine-learning framework at the base level, PaleoSTeHM will be built to take advantage of current and future computational advances without modifications to the user-facing product. It will facilitate the incorporation of complex likelihood structures, including the embedding of physical simulation models, and thus pave the way for further advances in paleo-modeling.

Please note that this repository is under active developmennt by Yucheng Lin, Alex Reed and Robert Kopp. If you have any questions please contact Yucheng: yc.lin@rutgers.edu. 

## Installation

```
git clone https://github.com/radical-collaboration/PaleoSTeHM.git
cd PaleoSTeHM/
pip install -r requirements.txt
```

PaleoSTeHM was developed under python version 3.7.3 and Jupyter Notebook 5.4.0. 

## File Descriptions
* **[Tutorials/PaleoSTeHM_Temporal.ipynb](Tutorials/PaleoSTeHM_Temporal.ipynb)** - A notebook contains tutorial about using Gaussian Process (GP) regression to infer temporal variation of sea-level change. It covers (1) Bayesian inference a single GP kernel using either Empirical Bayes or Fully Bayesian Analysis; (2) Bayesian inference with multiple GP kernels; (3) Bayesian inference incorporating temporal uncertainty using noisy-input Gaussian Process.   
* **[Tutorials/PaleoSTeHM_EIV_NI.ipynb](Tutorials/PaleoSTeHM_EIV_NI.ipynb)** - A notebook contains tutorial about different methods to incorporate temporal uncertainty in GP regression. Specifically, it focuses on two methods: (1) noisy-input Gaussian Process; (2) errors-in-variable Gaussian Process. 

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details
