.. _installation:

************
Installation
************
.. toctree::
    :maxdepth: 4

Local Installation
==================

1. **Clone the PaleoSTeHM repository**:

.. code-block:: none

   git clone https://github.com/radical-collaboration/PaleoSTeHM.git

2. **Create and activate a Python virtual environment**, and install PaleoSTeHM's Python dependencies in it. Using `conda <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment>`_:

.. code-block:: none

   cd PaleoSTeHM/
   conda env create -f environment.yml
   conda activate ve3PaleoSTeHM

3. **Load Jupyter Lab (or Jupyter Notebook)** and navigate to the tutorial of interest:

.. code-block:: none

   jupyter-lab

Google Colab Installation
=========================

The following are instructions on how to run PaleoSTeHM tutorials through `Google Colab <https://colab.research.google.com/>`_. One must have a Google account and access to Google Drive to use PaleoSTeHM in this configuration.

1. **Navigate to Google Colab** and select "+ New Notebook" when prompted.

2. **In the new notebook**, copy and paste the following into a cell:

.. code-block:: none

   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive
   !git clone https://github.com/radical-collaboration/PaleoSTeHM.git

3. **Open Google Drive** and navigate to the newly created PaleoSTeHM repo. Go to the tutorial you would like to run (i.e. /PaleoSTeHM/Tutorials/1.introduction/introduction.ipynb), right-click and select "run from Google Colab".

Pip Installation
=========================

.. code-block:: none

   pip install PaleoSTeHM