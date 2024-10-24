.. _tutorials:

Tutorial Contents
=================

Video Tutorial
--------------

- 2024 PaleoSTeHM workshop, in collaboration with PALSEA (PALeo constraints on SEA level rise) - `Day 1 <https://www.youtube.com/watch?v=OFkmNY6puh0&t=615s>`_, `Day 2 <https://www.youtube.com/watch?v=d9X5NnFHCwc>`_.


Introduction
------------

- **PaleoSTeHM background and data level modelling**: Introduction to Bayesian Hierarchical modelling, consisting of data level, process level, and parameter level modelling. This notebook will cover data level modelling with an example of building a data level model for coral-based sea-level records from the Great Barrier Reef.

  - `1. Introduction <https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F1.Introduction%2F1.Introduction.ipynb>`_ | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F1.Introduction%2F1.Introduction.ipynb)

- **Process level and parameter level modelling**: This tutorial covers process and parameter level modelling, including hand-drawn models, linear models, change-point models, Gaussian Process models, physical models, and mixed physical/statistical models.

  - `2. Process Level Modelling <https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F1.Introduction%2F2.Process_level_modelling.ipynb>`_ | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F1.Introduction%2F2.Process_level_modelling.ipynb)

- **Analysis Choice**: This tutorial discusses the difference between deterministic models and probabilistic (Bayesian) models, optimization methods for deterministic models (least squares and gradient-based optimization), and inference methods for Bayesian models, such as MCMC, Variational Bayes, and Empirical Bayes.

  - `3. Analysis Choice <https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F1.Introduction%2F3.Analysis_Choice.ipynb>`_ | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F1.Introduction%2F3.Analysis_Choice.ipynb)


Temporal Gaussian Process
--------------------------

- **Gaussian Process background**: A tutorial covering the basics of Gaussian Process, including calculating squared distance matrices, covariance matrices, hyperparameter definitions, conditional probability, and hyperparameter optimization.

  - `4. GP Background <https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F2.Temporal_GP%2F4.GP_background.ipynb>`_ | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F2.Temporal_GP%2F4.GP_background.ipynb)

- **Gaussian Process kernels and operations**: Introduction to popular GP kernels and instructions on how to combine different kernels.

  - `5. GP Kernels and Operations <https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F2.Temporal_GP%2F5.GP_kernels_and_operation.ipynb>`_ | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F2.Temporal_GP%2F5.GP_kernels_and_operation.ipynb)

- **Incorporating temporal uncertainty in Gaussian Process**: Two methods commonly used for Gaussian Process, including noisy-input GP and errors-in-variable GP.

  - `6. Temporal Uncertainty <https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F2.Temporal_GP%2F6.Temporal_uncer.ipynb>`_ | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F2.Temporal_GP%2F6.Temporal_uncer.ipynb)

- **Holocene sea-level analysis in New Jersey and Northern North Carolina**: This tutorial replicates Gaussian Process modelling results from New Jersey and Northern North Carolina as seen in Ashe et al., 2019.

  - `7. NJ & NNC RSL Analysis <https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F2.Temporal_GP%2F7.NJ_NNC_RSL.ipynb>`_ | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F2.Temporal_GP%2F7.NJ_NNC_RSL.ipynb)


Spatio-temporal Gaussian Process
--------------------------------

- **Spatio-temporal Gaussian Process Background**: Introduction to building a Spatio-temporal Gaussian Process model.

  - `8. STGP Background <https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F3.ST_GP%2F8.STGP_background.ipynb>`_ | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F3.ST_GP%2F8.STGP_background.ipynb)

- **Spatio-temporal Gaussian Process with multiple kernels**: Detailed instructions on kernel decomposition and building a Spatio-temporal GP model using multiple kernels.

  - `9. STGP Kernels <https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F3.ST_GP%2F9.STGP_kernels.ipynb>`_ | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F3.ST_GP%2F9.STGP_kernels.ipynb)

- **Spatio-temporal Gaussian Process with Physical Models**: Combining Spatio-temporal GP models with physical models.

  - `10. STGP with Physical Models <https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F3.ST_GP%2F10.STGP_with_physical_model.ipynb>`_ | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F3.ST_GP%2F10.STGP_with_physical_model.ipynb)


Applications
------------

- **Common Era Sea-Level Reconstruction**: Reconstructing Common Era global sea-level change using the method introduced in Kopp et al., 2016.

  - `11. Common Era Sea-Level Reconstruction <https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F4.Applications%2FKopp_et_al_2016.ipynb>`_ | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F4.Applications%2FKopp_et_al_2016.ipynb)

- **Updating Common Era Sea-Level Curve**: Add your own data to update the Common Era global sea-level curve, following the methodology of Walker et al., 2022.

  - `12. Update GMSL Curve <https://mybinder.org/v2/gh/radical-collaboration/PaleoSTeHM/HEAD?labpath=Tutorials%2F4.Applications%2FUpdating_GMSL_curve.ipynb>`
