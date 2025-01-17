import torch

class Constants:
    ############################################################################
    ###############################   OPTIONS   ################################
    ############################################################################
    results_dir_name = 'trained_models'
    store_wide_figures = True
    wide_figure_dimensions = (20,5)
    is_gpu_available = torch.cuda.is_available()
    enable_speedup = True # Enable only after model is debugged. All speedups based on https://www.youtube.com/watch?v=9mS1fIYj1So
    normalize_data = True # for new models used for the journal, the normalization shiould be enabled

    ############################################################################
    ######################   Datasets to train models   ########################
    ############################################################################
    datasets = [
        'FordB', # 92.86% 2 classes
        'FordA', # 96.54% 2 classes 
        'NonInvasiveFetalECGThorax1', # 42 Classes
        'NonInvasiveFetalECGThorax2', # 42 Classes
        'Wafer', # 99.98% - 2 classes
    ]

    ############################################################################
    ######################   Used model architectures   ########################
    ############################################################################
    network_architectures = [
        'Inception',
        'LSTM',
        'MLP',
        'ResNet',
        'ViT',
    ] # Architectures for which a model should be trained
    
    ############################################################################
    ###########################   Hyperparameters   ############################
    ############################################################################
    n_epochs = 500 # maximum number of epochs the network should be trained
    n_tries = 5 # number of tries that hyperopt tries to find the best hyperparameters
    # lr = 3e-4 # learning rate - if lr is set, than lr_min and lr_max are ignored
    lr_min = 1e-6 # minimum learning rate for hyperopt
    lr_max = 1e-2 # maximum learning rate for hyperopt
    bs = 256 # This batch size is used only for transformers ...  if bs is set, than bs_min, bs_max and bs_steps are ignored
    bs_min = 16 # minimum batch size for hyperopt
    bs_max = 32 # maximum batch size for hyperopt
    bs_steps = 6 # steps in which hyperopt changes the batch size
    min_loss_improvement = 0 # 0.0005 # for early stopping - minimum improvement of validation loss
    patience = 30 # for early stopping - if the validation loss hasn't improved for 'min_loss_improvement' in 'patience' epochs, the training stops
    model_save_name = 'model' # name of the stored model
    train_val_split_ratio = 0.8 # how many percent of all the training samples will be in the training set (rest goes into validation)
    n_seeds =  30
    save_only_best_model = True # If this is set to true, only the best auto_train will try to find the best model for a seed. Otherwise it stores the model as soon as the store_model_accuracy_threshold is achieved
    store_model_accuracy_threshold = 1

    ############### INTERPRET MODEL REGIONS ##################
    store_plots = False
