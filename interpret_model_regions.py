# This script calculates the MoRF and LeRF perturbation curves using region perturbation for all datasets and models specified in this script using the defined settings
# The results are then stored in a mongodb collection, defined in configs/dbconfig.py

import os
import csv, json
import time
import math
import copy
from datetime import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch

from configs.dbconfig import dbConfig
import pymongo

from utils.utils import *
from utils.networks import network_dict
from utils.constants import Constants as c

from sklearn import metrics

import warnings
warnings.simplefilter("ignore", UserWarning) # to ignore captum notification "warnings"

from interpretability_methods.interpretability_methods import *
from utils.subsequence_perturbation import SubSequencePerturber as ssp

from utils.utils import profile

enable_progress_prints = False

torch_device = torch.device('cuda' if c.is_gpu_available else 'cpu')

# TODO: Checklist: Correct results_collection selected in dbconfig.py?

# TODO: Checklist: use right experiment id
now = datetime.now()
timestamp = now.strftime('%Y_%m_%d__%H_%M_%S')
experiment_prefix = 'experiment_results'
experiment_id = '{}_{}'.format(experiment_prefix, timestamp)

# @profile # TODO: Checklist: Disable @profile
def interpret_model_regions():
    dbclient = pymongo.MongoClient(dbConfig.url)
    db = dbclient[dbConfig.dbName]
    results_collection = db[dbConfig.results_collection]

    if c.enable_speedup:
        torch.backends.cudnn.benchmark = True

    # TODO: Checklist: Correct datasets?
    # datasets
    data_folders = [
        'NonInvasiveFetalECGThorax1', # 94.68 % - 42 Classes
        'NonInvasiveFetalECGThorax2', # 96.68 % - 42 Classes
        'Wafer', # 99.98% - 2 classes
        'FordA', # 96.54% - 2 classes 
        'FordB', # 92.86% - 2 classes
    ]

    # TODO: Checklist: Enable all architectures
    network_names = [
        'ResNet',
        'Inception', 
        'LSTM', 
        'MLP', 
        'ViT'
    ] # Used for experiments

    # TODO: Checklist: Enable all attribution methods of interest
    # Model agnostic captum methods + custom "random" relevance methods
    interpretability_methods = [
        DeepLIFTCaptum,
        InputXGradientCaptum,
        RandomAttribution,
        GuidedBackpropCaptum,
        DeconvolutionCaptum,
        FeatureAblationCaptum,
        IntegratedGradientsCaptum,
        KernelShapCaptum,
        LimeCaptum,
        SaliencyCaptum,

        # GuidedGradCAMCaptum, # can only be used with CNNs
        # GradCAM, # can only be used with CNNs
        
    ]

    # TODO: Checklist: Enable all perturbation methods
    perturbation_methods = [
        ssp.UniformNoise100,
        ssp.UniformNoise50,
        ssp.UniformNoise25,
        ssp.SampleMean,
        ssp.Zero,

        ssp.SubsequenceMean,
        ssp.Inverse,
        ssp.Swap,

        ssp.OutOfDistHigh,
        ssp.OutOfDistLow,

        ssp.LinearInterpolation,
        ssp.QuadraticInterpolation,
        ssp.CubicInterpolation,
        ssp.Padding,
        ssp.Nearest,

        ssp.LeftNeighborWindow,
        ssp.RightNeighborWindow,
        ssp.NearestNeighborWindow,

        ssp.SimilarNeighborWindow,
        ssp.DissimilarNeighborWindow,
        ssp.GaussianBlur,
        ssp.SavitzkyGolay,
        ssp.Laplace,
    ]

    region_size_type = 'percent' # 'fixed' vs 'percent' - default is fixed

    # TODO: Checklist: Enable all region sizes
    # TODO: Add region size of 11 as fixed one 
    region_sizes = [
        2.5, # add per dataset how big the region size actually is
        10, # add per dataset how big the region size actually is
    ]

    # TODO: Checklist: Correct perturbation ratio?
    max_perturbation_ratio = 0.5 # How much percent of the output should be perturbed
    models_root_dir = 'models' # Root directory of trained models #TODO: Checklist: check directory for models
    # models_root_dir = 'best_models_journal' # Root directory of trained models
    # TODO: Checklist: High batch size?
    batch_size = 6200

    for folder_name in tqdm(data_folders, desc='{: <12}'.format('Datasets')):
        for network_name in tqdm(network_names, desc='{: <12}'.format('Models'), leave=False):
            
            network_dir = os.path.join(models_root_dir, folder_name, network_name)
            model_target_dirs = ['.'] # stay in same directory if multiple seeds not set
            model_target_dirs = list(os.listdir(network_dir))
            model_target_dirs.sort()

            # TODO: Checklist: Iterate over all seeds
            for seeded_model_name in tqdm(model_target_dirs, desc='{: <12}'.format('Seeds'), leave=False): # iterate over all models with different seeds
                
                model_dir = os.path.join(models_root_dir, folder_name, network_name, seeded_model_name)
                config_name = 'training_conf.json'
                path_to_config = os.path.join(model_dir, config_name)
                with open(path_to_config, 'r') as infile:
                    config = json.load(infile)

                network_arch = config['network_arch']
                is_univariate = config['is_univariate']
                dataset_name = config['dataset_name']
                model_save_name = config['model_save_name']

                if 'data_normalization_method' in config:
                    norm_type = config['data_normalization_method']
                    norm_params = config['normalization_parameters']
                else:
                    norm_type = None
                    if c.normalize_data:
                        norm_type = c.normalization_method
                    _, _, norm_params = load_univariate_UCR_dataset(dataset_name, normalization=norm_type)

                # Load test data
                if enable_progress_prints:
                    print('Load dataset "{}"'.format(dataset_name))
                if is_univariate:
                    test_x, test_y, _ = load_univariate_UCR_dataset(dataset_name, is_testset=True, normalization=norm_type, norm_params=norm_params)
                else:
                    test_x, test_y, _ = load_multivariate_UCR_dataset(dataset_name, is_testset=True, normalization=norm_type, norm_params=norm_params)
                
                # test_x = test_x[:batch_size] # TODO: COMMENT OUT THIS LINE!!! IT IS ONLY FOR DEBUGGING
                # test_y = test_y[:batch_size] # TODO: COMMENT OUT THIS LINE!!! IT IS ONLY FOR DEBUGGING

                unique_labels = np.unique(test_y)
                if np.min(unique_labels) > 0:
                    test_y -= np.min(unique_labels) # shift labels so they start with 0
                    unique_labels = np.unique(test_y) # after label shift

                input_length = test_x.shape[-1]
                n_channels = test_x.shape[1]
                n_outputs = len(unique_labels)

                # Load model
                if enable_progress_prints:
                    print('Load model: "{}", seed: "{}", batch size: {}'.format(network_name, seeded_model_name, batch_size))

                model = network_dict[network_arch](input_length, n_channels, n_outputs)
                state_dict = torch.load(os.path.join(model_dir,model_save_name + ".pt"))
                model.load_state_dict(state_dict)
                model.eval()

                if c.is_gpu_available:
                    model = model.cuda()

                for s_idx in tqdm(range(0, len(test_x), batch_size), desc='{: <12}'.format('Samples'), leave=False):
                    percent = s_idx / len(test_x) * 100
                    if enable_progress_prints:
                        print('Dataset: {} | Model: {} | Seed: {} | Completed: {:3.2f}%'.format(folder_name, network_arch, seeded_model_name, percent))

                    # get samples, labels and indeces
                    target_samples = np.array(test_x[s_idx:s_idx+batch_size], copy=True)
                    target_labels = test_y[s_idx:s_idx+batch_size]
                    sample_indexes = np.arange(s_idx, s_idx+len(target_samples))
                    
                    x_batch = torch.tensor(target_samples,dtype=torch.float32, device=torch_device).requires_grad_(True)

                    # make initial prediction
                    with torch.no_grad():
                        model_output = model(x_batch)

                    pred_indexes = torch.exp(model_output).argmax(dim=1)
                    if c.is_gpu_available:
                        predicted_labels = pred_indexes.cpu().numpy()
                        raw_output = torch.exp(model_output).squeeze().cpu().detach().numpy()
                    else:    
                        predicted_labels = pred_indexes.numpy()
                        raw_output = torch.exp(model_output).squeeze().detach().numpy()

                    if len(raw_output.shape) == 1:
                        raw_output = raw_output.reshape(1,-1)
                    probabilities_mask = np.zeros_like(raw_output)
                    for i in range(len(probabilities_mask)):
                        probabilities_mask[i][predicted_labels[i]] = 1
                    probabilities = raw_output[probabilities_mask == 1]

                    # OPTIONAL: Sift through predictions and remove the ones with a low probability and wrongly predicted class
                    remove_indexes = []
                    # for i in range(len(probabilities)):
                    #     target_label = target_labels[i]
                    #     predicted_label = predicted_labels[i]
                    #     probability = probabilities[i]*100

                    #     additional_text = []

                    #     if probability < probability_threshold:
                    #         additional_text.append('Prediction below probability threshold')

                    #     if predicted_label != target_label:
                    #         additional_text.append('Predicted wrong class')

                    #     if len(additional_text) > 0:
                    #         remove_indexes.append(i)

                    # Remove all samples that do not meet requirements
                    sample_indexes   = np.delete(sample_indexes,   remove_indexes, axis=0)
                    target_samples   = np.delete(target_samples,   remove_indexes, axis=0)
                    target_labels    = np.delete(target_labels,    remove_indexes, axis=0)
                    probabilities    = np.delete(probabilities,    remove_indexes, axis=0)
                    predicted_labels = np.delete(predicted_labels, remove_indexes, axis=0)
                    # x_batch = torch.tensor(target_samples,dtype=torch.float32).requires_grad_(True)
                    # if c.is_gpu_available:
                    #     x_batch = x_batch.cuda()

                    if len(sample_indexes) == 0:
                        continue

                    am_start_time = datetime.now()
                    for interpretability_method in tqdm(interpretability_methods, desc='{: <12}'.format('AMs'), leave=False):
                        torch.cuda.empty_cache()
                        try:
                            method_name = interpretability_method.__name__
                        except:
                            method_name = interpretability_method

                        if ((method_name == 'GradCAM') or (method_name == 'GuidedGradCAMCaptum')) and not ((network_name == 'ResNet') or (network_name == 'Inception')):
                            continue

                        if enable_progress_prints:
                            print('  Compute feature importance - method: "{}"'.format(method_name))

                        attr_method = interpretability_method(model)
                        attributions = torch.zeros_like(x_batch)
                        input_relevances = np.zeros((x_batch.shape[0],x_batch.shape[2]))

                        target = torch.tensor(predicted_labels, device=torch_device)

                        # debug_start_time = datetime.now()
                        torch.backends.cudnn.enabled = (network_name != 'LSTM') # captum cannot compute attributions for RNNs/LSTMs on the GPU, so we have to briefly disable cudnn
                        am_batch_size = 256
                        # attributions = attr_method.attribute(x_batch, target=target)
                        for i in range(0, x_batch.shape[0], am_batch_size):
                            end = i + am_batch_size
                            am_x_batch = x_batch[i:end]
                            am_target_batch = target[i:end]
                            attributions[i:end] = attr_method.attribute(am_x_batch,target=am_target_batch)
                            torch.cuda.empty_cache()

                        torch.backends.cudnn.enabled = True
                        if attributions.is_cuda:
                            attributions = attributions.cpu().detach().numpy()
                        else:
                            attributions = attributions.detach().numpy()
                        del target
                        torch.cuda.empty_cache()
                        # debug_end_time = datetime.now()
                        # debug_total = debug_end_time - debug_start_time
                        # print("--- AMs time: {:02d}:{:02d}:{:02d}".format((debug_total.seconds // 3600) % 24, (debug_total.seconds // 60) % 60, debug_total.seconds % 60))
                        # continue
                        
                        for i in range(len(x_batch)):
                            # --- NEW ---
                            attribution = attributions[i]  # .unsqueeze(0)
                            min_val = attribution.min()
                            max_val = attribution.max()
                            if (max_val - min_val) != 0:
                                attribution = (attribution - min_val) / float(max_val - min_val)
                            elif max_val != 0:
                                attribution = attribution / max_val
                            input_relevances[i] = attribution
                            
                        # Prepare results_obj - results are grouped by dataset, model, seed, sample, AM and contains the results using all region sizes and PMs
                        results_objs = []
                        for i in range(len(sample_indexes)):
                            s_id = int(sample_indexes[i])
                            target_label = int(target_labels[i])
                            input_relevance = input_relevances[i]

                            results_objs.append({
                                'dataset name': dataset_name,
                                'model architecture': network_arch,
                                'model seed': int(seeded_model_name.replace('seed_','')),
                                'experiment id': experiment_id,
                                'perturbation ratio': max_perturbation_ratio,
                                'models root dir': models_root_dir,
                                'class id': target_label,
                                'sample id': s_id,
                                'feature attribution method': method_name,
                                'input relevances': input_relevance.tolist(),
                                'results' : []
                            })

                        for region_size in tqdm(region_sizes, desc='{: <12}'.format('Region sizes'), leave=False):
                            if enable_progress_prints:
                                print('    Region size: {}'.format(region_size))
                            rs = region_size
                            if region_size_type == 'percent':
                                rs = math.ceil( (input_length * region_size) / 100)

                            # Average relevances in individual regions
                            relevance_orders = []

                            num_regions = math.ceil(input_length / rs)
                            perturbation_ratio_steps = math.ceil(((input_length / rs) * max_perturbation_ratio)) # max number of steps w.r.t. perturbation ratio
                            actual_num_of_perturbation_steps = min(perturbation_ratio_steps, num_regions) # needed for later, when actually perturbing the samples

                            for i in range(len(attributions)):
                                input_relevance = input_relevances[i]
                                
                                ss_relevances = []
                                for s in range(num_regions):
                                    ss_mean = input_relevance[s*rs:(s+1)*rs].mean()
                                    ss_relevances.append(ss_mean)
                                most_to_least_relevant_ss = np.flip(np.argsort(ss_relevances))
                                rel_points = ((most_to_least_relevant_ss + 1) * rs) - (rs // 2) - 1
                                rel_points[rel_points > (len(input_relevance)-1 )] = len(input_relevance)-1

                                rel_points = np.array(rel_points)

                                relevance_orders.append({
                                        'MoRF' : rel_points,
                                        'LeRF' : np.flip(np.array(rel_points, copy=True)),
                                })

                            for perturbation_method in tqdm(perturbation_methods, desc='{: <12}'.format('PMs'), leave=False):
                                if enable_progress_prints:
                                    print('      Perturbation Method: {}'.format(perturbation_method.__name__))
                                # prepare individual results entries that are associated with a sample and AM
                                for i in range(len(sample_indexes)):
                                    results_objs[i]['results'].append({
                                        'region size type' : region_size_type,
                                        'region size': rs,
                                        'perturbation method' : perturbation_method.__name__,
                                        'perturbation results': {
                                            'MoRF': [],
                                            'LeRF': []
                                        }
                                    })

                                for order in relevance_orders[0]:
                                    predictions = np.zeros((len(target_samples), actual_num_of_perturbation_steps+1)) # add 1 to number of regions to store initial prediction

                                    samples_batch = copy.deepcopy(target_samples)
                                    samples_tensor = torch.tensor(samples_batch,dtype=torch.float32, device=torch_device)
                                    # if c.is_gpu_available:
                                    #     samples_tensor = samples_tensor.cuda()

                                    with torch.no_grad():
                                        output = torch.exp(model(samples_tensor)).squeeze()

                                    if c.is_gpu_available:
                                        raw_output = output.cpu().numpy()
                                    else:
                                        raw_output = output.numpy()
                                    if len(raw_output.shape) == 1:
                                        raw_output = raw_output.reshape(1,-1)
                                    
                                    probabilities_mask = np.zeros_like(raw_output)
                                    for i in range(len(probabilities_mask)):
                                        probabilities_mask[i][predicted_labels[i]] = 1
                                    probabilities = raw_output[probabilities_mask == 1] * 100

                                    predictions[:,0] = probabilities

                                    # Procedure:
                                    #   1. Perturb the first step in all samples
                                    #   2. Make prediction on batch
                                    #   3. ...
                                    #   4. Profit!
                                    perturbers = []
                                    for i in range(len(samples_batch)):
                                        perturbers.append(perturbation_method(samples_batch[i][0]))

                                    for perturbation_step in range(actual_num_of_perturbation_steps):
                                        # Perturb the n-th step in all samples
                                        for i in range(len(samples_batch)):
                                            pos = relevance_orders[i][order][perturbation_step]

                                            if rs % 2 != 0:
                                                start = int(pos) - (rs // 2)
                                                end = int(pos) + (rs // 2)
                                            else:
                                                start = int(pos) - (rs // 2) + 1
                                                end = int(pos) + (rs // 2)


                                            if start < 0:
                                                start = 0
                                            if end >= input_length - 1:
                                                end = input_length - 1

                                            perturbed_sample = perturbers[i].perturb_subsequence(start, end + 1)
                                            perturbed_sample = np.expand_dims(perturbed_sample, axis=0)
                                            samples_batch[i] = perturbed_sample
                                        
                                        perturbed_batch = torch.tensor(samples_batch,dtype=torch.float32, device=torch_device)

                                        with torch.no_grad():
                                            output = torch.exp(model(perturbed_batch)).squeeze()
                                            torch.cuda.synchronize()

                                        if c.is_gpu_available:
                                            raw_output = output.cpu().numpy()
                                        else:
                                            raw_output = output.numpy()

                                        if len(raw_output.shape) == 1:
                                            raw_output = raw_output.reshape(1,-1)  

                                        probabilities_mask = np.zeros_like(raw_output)
                                        for i in range(len(probabilities_mask)):
                                            probabilities_mask[i][predicted_labels[i]] = 1
                                        probabilities = raw_output[probabilities_mask == 1] * 100

                                        predictions[:,perturbation_step+1] = probabilities # always +1 since the first position is the initial predictions

                                    del samples_batch
                                    del perturbed_sample
                                    del perturbed_batch

                                    for i in range(len(sample_indexes)):
                                        results_objs[i]['results'][-1]['perturbation results'][order] = predictions[i].tolist()
                        results_collection.insert_many(results_objs) # store results after all perturbation methods have been used for a sample and attribution method
                    am_end_time = datetime.now()
                    am_total = am_end_time - am_start_time
                    if enable_progress_prints:
                        print("--- AMs time: {:02d}:{:02d}:{:02d}".format((am_total.seconds // 3600) % 24, (am_total.seconds // 60) % 60, am_total.seconds % 60))
                    del x_batch

                del model

start_time = datetime.now()

interpret_model_regions()

print('Finished!')
end_time = datetime.now()
total_time = end_time - start_time

print("**   Evaluating attributions started at  : {}".format(start_time.strftime("%d.%m.%Y %H:%M:%S")))
print("**   Evaluating attributions finished at : {}".format(end_time.strftime("%d.%m.%Y %H:%M:%S")))
print("**   ")
print("**** Total time for evaluating attributions took: {} day(s) and {:02d}:{:02d}:{:02d}".format(total_time.days, (total_time.seconds // 3600) % 24, (total_time.seconds // 60) % 60, total_time.seconds % 60))