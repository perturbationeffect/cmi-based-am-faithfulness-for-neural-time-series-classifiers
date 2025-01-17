# This script automatically trains multiple models for multiple models.
# The exact training settings (e.g., which datasets, which model architectures, epochs, etc.) used for training are provided in utils/constants.py

import os
import json
from datetime import datetime
import numpy as np
import torch
from sklearn.utils import shuffle

from utils.utils import *
from utils.networks import network_dict
from train import tpe_search, evaluate_model

import matplotlib.pyplot as plt

from utils.constants import Constants as c

print("Training on GPU: ", c.is_gpu_available)

start_time = datetime.now()

if c.enable_speedup:
    torch.backends.cudnn.benchmark = True

for dataset_name in c.datasets:
    for network_arch in c.network_architectures:
        # for seed in c.seeds:
        for seed in range(c.n_seeds):
            ############################################
            # Auto-train config
            is_univariate = True
            
            results_dir = os.path.join(c.results_dir_name, dataset_name, network_arch, 'seed_{}'.format(seed) )
            ############################################

            print('************************** Start auto_train **************************')
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            print('* Univariate: "{}"'.format(is_univariate))
            print('* Dataset: "{}"'.format(dataset_name))
            print('* Network: "{}"'.format(network_arch))
            print('* Results_dir: "{}"'.format(results_dir))
            print('**********************************************************************')

            # define if data should be normalized and how
            norm_type = None
            if c.normalize_data:
                norm_type = c.normalization_method

            # load data
            print('Load dataset "{}"'.format(dataset_name))
            if is_univariate:
                samples, labels, norm_params = load_univariate_UCR_dataset(dataset_name, normalization=norm_type)
            else:
                # It has to be tested if load_multivariate_UCR_dataset is still compatible with auto model training
                samples, labels, norm_params = load_multivariate_UCR_dataset(dataset_name, normalization=norm_type)

            if np.isnan(samples).any() or np.isnan(labels).any():
                print('Skipping dataset "{}" since it contains NaNs'.format(dataset_name))
                continue

            # check if labels start with zero
            unique_labels = np.unique(labels)
            if min(unique_labels) > 0:
                labels -= min(unique_labels) # labels should start with 0

            # shuffle data 
            samples, labels = shuffle(samples, labels, random_state=seed)

            # Split into training and validation set by maintaing the ration of class samples in the train and validation set
            print("unique_labels: ", len(unique_labels))
            print("")
            for i in range(len(unique_labels)):
                class_split = int(c.train_val_split_ratio * len(np.where(labels == i)[0]))
                if i == 0:
                    train_x = samples[np.where(labels == i)][:class_split]
                    train_y = labels[np.where(labels == i)][:class_split]
                    val_x = samples[np.where(labels == i)][class_split:]
                    val_y = labels[np.where(labels == i)][class_split:]
                else:
                    train_x = np.concatenate([train_x, samples[np.where(labels == i)][:class_split]])
                    train_y = np.concatenate([train_y, labels[np.where(labels == i)][:class_split]])
                    val_x = np.concatenate([val_x, samples[np.where(labels == i)][class_split:]])
                    val_y = np.concatenate([val_y, labels[np.where(labels == i)][class_split:]])

            train_x, train_y = shuffle(train_x, train_y, random_state=seed)    
            val_x, val_y = shuffle(val_x, val_y, random_state=seed)


            input_length = train_x.shape[-1]
            n_channels = samples.shape[1]
            n_outputs = len(np.unique(labels))

            print("train_x: ", train_x.shape)
            print("train_y: ", train_y.shape)
            print("val_x: ", val_x.shape)
            print("val_y: ", val_y.shape)

            for i in range(len(unique_labels)):
                print("train class {} - (%): {:.2f}".format(i, len(np.where(train_y == i)[0]) / len(train_y)))
                print("val   class {} - (%): {:.2f}".format(i, len(np.where(val_y == i)[0]) / len(val_y)))

            print("min_value: ", samples.min())
            print("max_value: ", samples.max())
            print("#channels: ", n_channels)
            print("#outputs: ", n_outputs)

            ########################### Train model ###########################

            # Create results dir if it does not exist
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            criterion = torch.nn.NLLLoss() # Loss function

            training_conf = {
                'network_name' : network_arch,
                'criterion' : criterion,
                'seed' : seed,
                'train_x' : train_x,
                'train_y' : train_y,
                'val_x' : val_x,
                'val_y' : val_y,
                'results_dir' : results_dir
            }

            status = tpe_search(training_conf)

            if len(os.listdir(results_dir)) == 0:
                os.rmdir(results_dir)
                continue

            # save plots of losses
            plt.plot(status.best_train_losses, color='orange', label='train loss', )
            plt.plot(status.best_valid_losses, color='blue', label='valid loss')

            plt.legend()
            plt.axvline(x=status.epoch_model_saved - 1, color='red', label='model saved') # -1 since the losses are 0 indexed
            plt.savefig(os.path.join(results_dir, 'losses_with_savepoint.png'))

            plt.close()

            ########################### Evaluate model ###########################

            # Load model
            model = network_dict[network_arch](input_length, n_channels, n_outputs)
            state_dict = torch.load(os.path.join(results_dir,c.model_save_name + ".pt"))
            model.load_state_dict(state_dict)

            # Load test data
            print('Load test dataset "{}"'.format(dataset_name))
            if is_univariate:
                test_x, test_y, _ = load_univariate_UCR_dataset(dataset_name, is_testset=True, normalization=norm_type, norm_params=norm_params)
                # test_x, test_y, _ = load_ecg_heartbeat_categorization_dataset('mitbih', is_testset=True, normalization=norm_type, norm_params=norm_params)
                # test_x, test_y, _ = load_heartbeat_sound_dataset(is_testset=True, normalization=norm_type, norm_params=norm_params)
                
            else:
                test_x, test_y, _ = load_multivariate_UCR_dataset(dataset_name, is_testset=True, normalization=norm_type, norm_params=norm_params)

            # check if starts with zero
            unique_labels = np.unique(test_y)
            if min(unique_labels) > 0:
                test_y -= min(unique_labels)

            # Evaluate
            status.best_test_loss, status.best_acc_test, status.best_f1_test = evaluate_model(model, criterion, test_x, test_y)

            # save training config
            model_config = {
                'best_test_loss' : status.best_test_loss,
                'best_f1_test' : status.best_f1_test,
                'best_acc_test' : status.best_acc_test,
                'best_valid_loss' : status.best_valid_loss,
                'best_f1_valid' : status.best_f1_valid,
                'model_save_name' : c.model_save_name,
                'is_univariate' : is_univariate,
                'dataset_name' : dataset_name,
                'network_arch' : network_arch,
                'seed' : seed,
                'best_train_loss' : status.best_train_loss,
                'batch_size' : status.best_bs,
                'learning_rate' : status.best_lr,
                'epochs' : status.n_epochs,
                'patience' : status.patience,
                'min_loss_improvement' : status.min_loss_improvement,
                'epoch_model_saved' : status.epoch_model_saved,
                'optimizer' : status.optimizer,
                'criterion' : status.criterion,
                'optimization_losses' : status.optimization_losses,
                'best_train_losses' : status.best_train_losses,
                'best_valid_losses' : status.best_valid_losses,
                'data_normalization_method' : norm_type,
                'normalization_parameters' : norm_params,
            }

            with open(os.path.join(results_dir, 'training_conf.json'), 'w') as outfile:
                json.dump(model_config, outfile, indent=2)

            print('\n\n')
            print('Finished!')
            status.print_summary()
            print("Trainable parameters: ", count_parameters(model))

end_time = datetime.now()
total_time = end_time - start_time

print ("Training started at  : {}".format(start_time.strftime("%d.%m.%Y %H:%M:%S")))
print ("Training finished at : {}".format(end_time.strftime("%d.%m.%Y %H:%M:%S")))

print("**** Total time for training took: {} day(s) and {:02d}:{:02d}:{:02d}".format(total_time.days, (total_time.seconds // 3600) % 24, (total_time.seconds // 60) % 60, total_time.seconds % 60))