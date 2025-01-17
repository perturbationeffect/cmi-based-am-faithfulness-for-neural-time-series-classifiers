# CMI-based Feature Attribution Validation for Neural Time Series Classifiers
This is the supporting code for the paper "A Comprehensive Analysis of Perturbation Methods in Explainable AI Feature Attribution Validation for Neural Time Series Classifiers" by Simic et al.

This code was used to train the models and evaluate the included attribution methods (AMs) and perturbation methods (PMs) on the trained models.
Moreover, scripts and notebooks for results analysis are included as well.

# Pre-requisites
- Anaconda package manager. The required libraries for the environment can be installed via the `install_conda_dependencies.sh` script.
- MongoDB should be installed and running to store the computed explanations and perturbation curves.
- Unzip the FordA and FordB datasets under `data/UCR_datasets/Univariate`

# File descriptions
- `configs/dbconfig.py` - should contain the configuration for accessing the mongodb, which is used to store all attributions and morf/lerf curve values for all perturbation methods
- `interpretability_methods/interpretability_methods.py` contains implementations and helper functions of investigated attribution methods
- `auto_train.py` - automatically trains models using the settings provided in `utils/constants.py`
- `utils` - contains implementations of neural networks, helper functions, as well as perturbation method implementations
- `find_best_model_seeds.py` - identifies model seeds of a dataset and model architecture with best test loss - only necessary if many models have been trained
- `interpret_model_regions.py` - applies the attribution methods and perturbation methods over the samples of the test set using the provided models
- `train.py` - implements the actual training loop used by `auto_train.py`
- `install_conda_dependencies.sh` - install all the necessary packages in the currently active conda environment
- `results_analysis.py` - uses computed morf and lerf curves to compute DDS and PES metrics
- `notebook - results summary and figures.ipynb` calculates the CMIs based on the PES and DDS values and generates figures
- `notebook - CMI-based AM evaluation pipeline.ipynb` loads a single provided model and dataset and applies the complete pipeline. NOTE: It can take very long to run it on many datasets and models

# Order in which the files should be run
`auto_train.py` is used to train the models automatically for the provided datasets

`interpret_model_regions.py` computes all explanations and runs region perturbation with all perturbation methods to compute and store the MoRF and LeRF perturbation curve values in a MongoDB collection

`results_analysis.py` computes DDS and PES scores from the MoRF and LeRF curves in the MongoDB collection and produces a .csv file, which contains the scores for each dataset, model, am, pm and region size combination, over all samples and split by class

`notebook - results summary and figures.ipynb` is then used to calculate the CMIs based on the PES and DDS values, perform the weighted ranking and generate figures