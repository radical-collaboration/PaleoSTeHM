.. _tutorials:

Tutorial Contents
=================

Video Tutorial
--------------

- 2024 PaleoSTeHM workshop, in collaboration with PALSEA (PALeo constraints on SEA level rise) - `Day 1 <https://www.youtube.com/watch?v=OFkmNY6puh0&t=615s>`_, `Day 2 <https://www.youtube.com/watch?v=d9X5NnFHCwc>`_.


Introduction
------------

- `PaleoSTeHM background and data level modelling <https://github.com/radical-collaboration/PaleoSTeHM/blob/main/Tutorials/1.Introduction/1.Introduction.ipynb>`_: Introduction to Bayesian Hierarchical modelling, consisting of data level, process level, and parameter level modelling. This notebook will cover data level modelling with an example of building a data level model for coral-based sea-level records from the Great Barrier Reef.

- `Process level and parameter level modelling <https://github.com/radical-collaboration/PaleoSTeHM/blob/main/Tutorials/1.Introduction/2.Process_level_modelling.ipynb>`_: This tutorial covers process and parameter level modelling, including hand-drawn models, linear models, change-point models, Gaussian Process models, physical models, and mixed physical/statistical models.

- `Analysis Choice <https://github.com/radical-collaboration/PaleoSTeHM/blob/main/Tutorials/1.Introduction/3.Analysis_Choice.ipynb>`_: This tutorial discusses the difference between deterministic models and probabilistic (Bayesian) models, optimization methods for deterministic models (least squares and gradient-based optimization), and inference methods for Bayesian models, such as MCMC, Variational Bayes, and Empirical Bayes.


Temporal Gaussian Process
--------------------------

- `Gaussian Process background <https://github.com/radical-collaboration/PaleoSTeHM/blob/main/Tutorials/2.Temporal_GP/4.GP_background.ipynb>`_: A tutorial covering the basics of Gaussian Process, including calculating squared distance matrices, covariance matrices, hyperparameter definitions, conditional probability, and hyperparameter optimization.

- `Gaussian Process kernels and kernel operations <https://github.com/radical-collaboration/PaleoSTeHM/blob/main/Tutorials/2.Temporal_GP/5.GP_kernels_and_operation.ipynb>`_: Introduction to popular GP kernels and instructions on how to combine different kernels.

- `Incorporate temporal uncertainty in Gaussian Process <https://github.com/radical-collaboration/PaleoSTeHM/blob/main/Tutorials/2.Temporal_GP/6.Temporal_uncer.ipynb>`_: Two methods commonly used for Gaussian Process, including noisy-input GP and errors-in-variable GP.

- `Holocene sea-level analysis in New Jersey and Northern North Carolina <https://github.com/radical-collaboration/PaleoSTeHM/blob/main/Tutorials/2.Temporal_GP/7.NJ_NNC_RSL.ipynb>`_: This tutorial replicates Gaussian Process modelling results from New Jersey and Northern North Carolina as seen in Ashe et al., 2019.

Spatio-temporal Gaussian Process
--------------------------------

- `Spatio-temporal Gaussian Process Background  <https://github.com/radical-collaboration/PaleoSTeHM/blob/main/Tutorials/3.ST_GP/8.STGP_background.ipynb>`_: Introduction to building a Spatio-temporal Gaussian Process model.

- `Spatio-temporal Gaussian Process with multiple kernels <https://github.com/radical-collaboration/PaleoSTeHM/blob/main/Tutorials/3.ST_GP/9.STGP_kernels.ipynb>`_ Detailed instructions on kernel decomposition and building a Spatio-temporal GP model using multiple kernels.

- `Spatio-temporal Gassusian Process with Physical Models <https://github.com/radical-collaboration/PaleoSTeHM/blob/main/Tutorials/3.ST_GP/10.STGP_with_physical_model.ipynb>`_: Combining Spatio-temporal GP models with physical models.

Applications
------------

- `Common Era Sea-Level Reconstruction <https://github.com/radical-collaboration/PaleoSTeHM/blob/main/Tutorials/4.Applications/Kopp_et_al_2016.ipynb>`_: Reconstructing Common Era global sea-level change using the method introduced in Kopp et al., 2016.

- `Updating Common Era sea-level curve <https://github.com/radical-collaboration/PaleoSTeHM/blob/main/Tutorials/4.Applications/Updating_GMSL_curve.ipynb>`_: Add your own data to update the Common Era global sea-level curve, following the methodology of Walker et al., 2022.

PaleoSTeHM User Interface
------------

- `Automatic Spatiotemporal sea-level analysis <https://github.com/radical-collaboration/PaleoSTeHM/blob/main/PaleoSTeHM_UI/Holocene_Spatiotemporal_analysis/Holocene_SP_anlysis.ipynb>`_: Automatic implementation/optimization/visulization of Holocene sea-level change using a spatiotemporal Gaussian Process model with a zero mean function.
