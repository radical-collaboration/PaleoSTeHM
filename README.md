# PaleoSTeHM - A modern, scalable Spatio-Temporal Hierarchical Modeling framework for paleo-environmental data


<a href="https://paleostehm.org">
    <img src="Docs/PSTHM_logo.png" alt="Project Logo" width="500">
</a>

By [Rutgers Earth System Science & Policy Lab](https://earthscipol.net/)

## Project overview
This repository contains the Jupyter Notebook based tutorials for PaleoSTeHM project, which will develop a framework for spatiotemporal hierarchical modeling of paleodata that builds upon modern, scalable software infrastructure for machine learning. By leveraging an existing, widely used machine-learning framework at the base level, PaleoSTeHM will be built to take advantage of current and future computational advances without modifications to the user-facing product. It will facilitate the incorporation of complex likelihood structures, including the embedding of physical simulation models, and thus pave the way for further advances in paleo-modeling. 

PaleoSTeHM is a paleo-environmental modelling faced API based upon [pyro](https://pyro.ai/), a universal probabilistic programming language supported by [PyTorch](https://pytorch.org/) on the backend. Therefore, it not only supports probabilistic programming not also the auto-differentiation and GPU acceleration. This API will provide easy-to-use models to infer the spatio or spatio-temporal variation of environmental change from geological data, with tutorials that cover the background theories and hand-on practicals. 

The work described in this repository is documented in the following publication:  
**DOI:** [10.5194/egusphere-2024-2183](https://doi.org/10.5194/egusphere-2024-2183)


Please note that this repository is managed by Yucheng Lin, Alex Reedy and Robert Kopp. If you have any questions please contact Yucheng: yc.lin@rutgers.edu. 


## Installation

### Option 1: Full Local Installation (Includes Tutorials)

1. Clone the PaleoSTeHM repository.

```
git clone https://github.com/radical-collaboration/PaleoSTeHM.git
```

2. Create and activate a Python virtual environment, and install PaleoSTeHM's Python 
dependencies in it. Using [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment):

```
cd PaleoSTeHM/
conda env create -f environment.yml
conda activate ve3PaleoSTeHM
```

3. Load Jupyter Lab (or Jupyter Notebook) and navigate to the tutorial of interest.

```
jupyter-lab
```

### Option 2: Google Colab Installation (Includes Tutorials)

The following are instructions on how to run PaleoSTeHM tutorials through Google Colab (https://colab.research.google.com/). One must have a Google account and access to Google Drive to use PaleoSTeHM in this configuration.

1. Navigate to Google Colab and select "+ New Notebook" when prompted.

2. In the new notebook copy and paste the following into cell.

```
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive
!git clone https://github.com/radical-collaboration/PaleoSTeHM.git
```

3. Open Google Drive and navigate to the newly created PaleoSTeHM repo. Go to the tutorial you would like to run (i.e. /PaleoSTeHM/Tutorials/1.introduction/introduction.ipynb) right click and select "run from Google Colab".


### Option 3: Lightweight Local Installation (Code Only)

```
pip install PaleoSTeHM==1.0
```


## Tutorial Contents
### Video Tutorial
  - 2024 PaleoSTeHM workshop, in collaboration with PALSEA (PALeo constraints on SEA level rise) - [Day 1](https://www.youtube.com/watch?v=OFkmNY6puh0&t=615s), [Day 2](https://www.youtube.com/watch?v=d9X5NnFHCwc). 

### Introduction
  - **[PaleoSTeHM background and data level modelling](Tutorials/1.Introduction/1.Introduction.ipynb)** - Introduction to Bayesian Hierachical modelling, which consist of data level, process level and parameter level modelling. This notebook will cover data level modelling with an illustrative example of building data level model for coral-based sea-level records from the Great Barrier Reef. [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F1.Introduction%2F1.Introduction.ipynb)
  - **[Process level and parameter level modelling](Tutorials/1.Introduction/2.Process_level_modelling.ipynb)** - Process level and parameter level modelling tutorial, which will cover a range of process level model for paleo-environmental research, including hand-drawn model, linear model, change-point model, Gaussian Process model, physical model, and mixed physical and statistical model. [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F1.Introduction%2F2.Process_level_modelling.ipynb)
  - **[Analysis Choice](Tutorials/1.Introduction/3.Analysis_Choice.ipynb)** - Analysis choice tutorial, which will cover the difference between a deterministic model and a probablistic model (we will mostly use a Bayesian model), the optimization methods for deterministic model (least sqaures and gradient-based optimization) and inference method for Bayesian model, including Markov Chain Monte Carlo, Variational Bayes and Empirical Bayes. [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F1.Introduction%2F3.Analysis_Choice.ipynb)

### Temporal Gaussian Process
  - **[Gaussian Process background](Tutorials/2.Temporal_GP/4.GP_background.ipynb)** - Gaussian Process background tutorial, which covers the basics of Gaussian Process, including calculating squared distance matrix and covariance matrix, hyperparameter definitions and their impact, conditional probability and hyperparameter optimization. [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F2.Temporal_GP%2F4.GP_background.ipynb)
  - **[Gaussian Process kernels and kernel operations](Tutorials/2.Temporal_GP/5.GP_kernels_and_operation.ipynb)** - Gaussian Process kernel tutorial, which introduces some popular GP kernels and instructions on how to combine different kernels. [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F2.Temporal_GP%2F5.GP_kernels_and_operation.ipynb)
  - **[Incorporate temporal uncertainty in Gaussian Process](Tutorials/2.Temporal_GP/6.Temporal_uncer.ipynb)**  - This tutorial introduce two commonly-used methods for Gaussian Process: noisy-input Gaussian Process and errors-in-variable Gaussian Process. [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F2.Temporal_GP%2F6.Temporal_uncer.ipynb)
  - **[Holocene sea-level analysis in New Jersey and Northern North Carolina](Tutorials/2.Temporal_GP/7.NJ_NNC_RSL.ipynb)**  - This tutorial will use the contents covered above in order to replicate Gaussian Process modelling results in New Jersey and Northern North Carolina as in [Ashe et al., 2019](https://www.sciencedirect.com/science/article/abs/pii/S0277379118302130). [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F2.Temporal_GP%2F7.NJ_NNC_RSL.ipynb)

### Spatio-temporal Gaussian Process 
 - **[Spatio-temporal Gaussian Process Background ](Tutorials/3.ST_GP/8.STGP_background.ipynb)** - Background of building a Spatio-temporal Gaussian Process model. [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F3.ST_GP%2F8.STGP_background.ipynb)
 - **[Spatio-temporal Gaussian Process with multiple kernels](Tutorials/3.ST_GP/9.STGP_kernels.ipynb)** - Spatio-temporal Gaussian Process with multiple kernels with instructions on kernel decomposition.  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F3.ST_GP%2F9.STGP_kernels.ipynb)
 - **[Spatio-temporal Gassusian Process with Physical Models](Tutorials/3.ST_GP/10.STGP_with_physical_model.ipynb)** - Spatio-temporal Gaussian Process in combination with one or multiple physical models. [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F3.ST_GP%2F10.STGP_with_physical_model.ipynb)

### Applications
 - **[Common Era Sea-Level Reconstruction](Tutorials/4.Applications/Kopp_et_al_2016.ipynb)** - Reconstructing Common Era global sea-level change using method introduced in [Kopp et al., 2016](https://www.pnas.org/doi/abs/10.1073/pnas.1517056113). [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F4.Applications%2FKopp_et_al_2016.ipynb).

  - **[Updating Common Era sea-level curve](Tutorials/4.Applications/Updating_GMSL_curve.ipynb)** - Updating Common Era global sea-level change by adding your own data along with [Walker et al., 2022](https://www.nature.com/articles/s41467-022-28564-6).

## PaleoSTeHM User Interface 
- **[Automatic Spatiotemporal sea-level analysis](PaleoSTeHM_UI/Holocene_Spatiotemporal_analysis/Holocene_SP_anlysis.ipynb)** - Automatic implementation/optimization/visulization of Holocene sea-level change using a spatiotemporal Gaussian Process model with a zero mean function.


## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details

## Acknowledgements

This work is supported by the National Science Foundation under awards 2002437 and 2148265. The opinions, findings, and conclusions or recommendations expressed are those of the authors and do not necessarily reflect the views of the National Science Foundation.
