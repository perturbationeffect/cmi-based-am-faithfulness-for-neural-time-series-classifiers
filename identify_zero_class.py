# This script identifies the class which the model predicts when the input gets out of distribution.
# This basically works works by drastically perturbing the input and observing which class the model predicts

import os
import time
import random

import numpy as np
import torch

from configs.dbconfig import dbConfig
import pymongo

from utils.utils import *
from utils.networks import network_dict
from utils.constants import Constants as c

import warnings
warnings.simplefilter("ignore", UserWarning) # to ignore captum notification "warnings"

from interpretability_methods.interpretability_methods import *


from utils.subsequence_perturbation import SubSequencePerturber as ssp

from functools import lru_cache

DISABLE_PRINTS = True

# Returns the zero class label
@lru_cache(maxsize=None)
def identify_zero_class(dataset_name, network_arch, seed):
    model_root_dir = 'models'
    model_dir = os.path.join(model_root_dir, dataset_name, network_arch, 'seed_{}'.format(seed))
    config_name = 'training_conf.json'
    path_to_config = os.path.join(model_dir, config_name)
    with open(path_to_config, 'r') as infile:
        config = json.load(infile)

    perturbation_method = ssp.OutOfDistHigh
    window_size = 7

    max_valid_samples = 5

    # Load dataset
    if 'data_normalization_method' in config:
        norm_type = config['data_normalization_method']
        norm_params = config['normalization_parameters']
    else:
        norm_type = None
        if c.normalize_data:
            norm_type = c.normalization_method
        _, _, norm_params = load_univariate_UCR_dataset(dataset_name, normalization=norm_type)

    test_x, test_y, _ = load_univariate_UCR_dataset(dataset_name, is_testset=True, normalization=norm_type, norm_params=norm_params)

    unique_labels = np.unique(test_y)
    if np.min(unique_labels) > 0:
        test_y -= np.min(unique_labels) # shift labels so they start with 0
        unique_labels = np.unique(test_y) # after label shift

    input_length = test_x.shape[-1]
    n_channels = test_x.shape[1]
    n_outputs = len(unique_labels)

    # Load model
    

    # print('Load model: "{}", seed: "{}"'.format(network_arch, seed))
    model = network_dict[network_arch](input_length, n_channels, n_outputs)
    state_dict = torch.load(os.path.join(model_dir, "model.pt"))
    model.load_state_dict(state_dict)
    model.eval()

    class_results = {}

    for label in unique_labels:
        class_samples = test_x[test_y == label]
        class_results[int(label)] = 0

        valid_sample_counter = 0
        for sample in class_samples:
            if valid_sample_counter >= max_valid_samples:
                break

            # convert sample to tensor
            target_sample = np.array(sample, copy=True)
            x_batch = torch.tensor(target_sample,dtype=torch.float32).unsqueeze(0).requires_grad_(True)

            # make initial prediction
            with torch.no_grad():
                model_output = model(x_batch)
                torch.cuda.empty_cache()

            # get class score
            pred_index = torch.exp(model_output).argmax(dim=1)
            if c.is_gpu_available:
                predicted_label = pred_index.cpu().numpy()
                original_probability = torch.exp(model_output).squeeze().cpu().detach().numpy()[predicted_label][0]
            else:    
                predicted_label = pred_index.numpy()
                original_probability = torch.exp(model_output).squeeze().detach().numpy()[predicted_label][0]

            if not DISABLE_PRINTS:
                print('  actual label            : {: 2d}'.format(int(label)))
                print('  predicted label         : {: 2d}'.format(int(predicted_label)))
                print('  original probability    : {:.2f}%'.format(float(original_probability*100)))


            # Skip the sample if the model has a low prediction score for this particular sample
            if original_probability < 0.95:
                if not DISABLE_PRINTS:
                    print('WARNING! Class score is below 95%... ')
                continue

            # Skip the sample if the model predicted the wrong class
            if predicted_label != label:
                if not DISABLE_PRINTS:
                    print('ERROR: Model predicted wrong class... ')
                continue

            valid_sample_counter += 1
            
            perturber = perturbation_method(target_sample[0])

            start = random.randint(0,input_length - window_size)
            end = start + window_size

            perturbed_sample = perturber.perturb_subsequence(start, end + 1)
            perturbed_sample = np.expand_dims(perturbed_sample, axis=0)
            perturbed_batch = torch.tensor(perturbed_sample,dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                output = torch.exp(model(perturbed_batch)).squeeze()
            new_probability = output[int(label)].item()

            if not DISABLE_PRINTS:
                print('  new probability         : {:.2f}%'.format(float(new_probability*100)))

            if new_probability >= .99:
                class_results[int(label)] += 1

    
    res_list = np.array([class_results[k] for k in class_results])

    if __name__ == "__main__":
        missing_classes = list(set(map(lambda x : int(x), unique_labels)) - set(class_results.keys()))
        print('Missing classes: ', missing_classes)
        print('Probability increased over 99% afer perturbation')
        for label in class_results:
            print('  {} : {}'.format(label, class_results[label]))

        if len(res_list[res_list == 0]) == len(res_list):
            print('No zero class identified')
        else:
            print('Zero class: {}'.format(int(res_list.argmax())))
        print(res_list)


    if len(res_list[res_list == 0]) == len(res_list):
        return None
    else:
        return int(res_list.argmax())
