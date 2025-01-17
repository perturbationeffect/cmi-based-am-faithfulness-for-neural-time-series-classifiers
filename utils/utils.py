import os
import json
import numpy as np
import pandas as pd
from .constants import Constants as c
import torch
import cProfile, pstats, io

from scipy.io import arff
from scipy.io.wavfile import read

import matplotlib.pyplot as plt

def list_univariate_UCR_datasets():
    dataset_parent_dir_path = 'data/UCR_datasets/Univariate'
    datasets = [d for d in os.listdir(dataset_parent_dir_path) if os.path.isdir(os.path.join(dataset_parent_dir_path, d))]
    return datasets


def list_multivariate_UCR_datasets():
    dataset_parent_dir_path = 'data/UCR_datasets/Multivariate'
    datasets = [d for d in os.listdir(dataset_parent_dir_path) if os.path.isdir(os.path.join(dataset_parent_dir_path, d))]
    return datasets


def load_UCR_arff_dataset_from_path(path, is_testset=False):
    samples = None
    labels = None

    dataset_parent_dir_path = path
    extension = '.arff'
    if is_testset:
        suffix = '_TEST'
    else:
        suffix = '_TRAIN'

    name = os.path.basename(dataset_parent_dir_path)
    dataset_path = os.path.join(dataset_parent_dir_path, name + suffix + extension)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError('A dataset with the name "{}" does not exist under the path "{}"'.format(name, dataset_parent_dir_path))

    data, meta = arff.loadarff(dataset_path)
    # Examples of structure of data
    # data[0] <- sample 1, data + labels
    # data[0][1] <- sample 1, label
    # data[0][0][0] <- sample 1, data of first channel
    # data[0][0][1] <- sample 1, data of second channel

    root = np.array(data[:].tolist())

    data = root[:,0]
    data_channels = np.array(data[:].tolist())
    samples = np.array(data_channels[:].tolist())

    labels = root[:,1]
    try:
        float(labels[0])
        labels = labels.astype('float')
    except:
        labels = labels.astype('str')
    
    return samples, labels


def normalize_samples(samples, normalization = None, norm_params = None):
    if normalization == 'standard':
        if norm_params is None:
            norm_params = {
                'mean' : samples.mean(),
                'std' : samples.std(),
            }
        samples = (samples - norm_params['mean']) / norm_params['std']

    elif normalization == 'minmax': # [-1,1]
        if norm_params is None:
            norm_params = {
                'min' : samples.min(),
                'max' : samples.max(),
            }
        samples = (2 * ( (samples - norm_params['min']) / (norm_params['max'] - norm_params['min']) ) ) - 1

    elif normalization == 'sample_minmax':
        for i in range(len(samples)):
            sample = samples[i]
            samples[i] = (2 * ( (sample - sample.min()) / (sample.max() - sample.min()) ) ) - 1

    return samples, norm_params


def load_univariate_UCR_dataset(name, is_testset=False, normalization = None, norm_params = None):
    '''Loads the raw data of one of the UCR univariate datasets

    Args:
        name (str): directory name of the dataset
        is_testset (bool): if set to True, the testset data will be loaded instead of the training set (default False)

    Returns:
        samples (list)
        labels (list)

    Raises:
        FileNotFoundError: If the dataset that has to be loaded does not exist

    Datasets downloaded from: http://www.timeseriesclassification.com/Downloads/Archives/Univariate2018_arff.zip
    '''
    samples = None
    labels = None

    dataset_parent_dir_path = 'data/UCR_datasets/Univariate'
    extension = '.txt'
    if is_testset:
        suffix = '_TEST'
    else:
        suffix = '_TRAIN'
    dataset_path = os.path.join(dataset_parent_dir_path, name, name + suffix + extension)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError('A dataset with the name "{}" does not exist under the path "{}"'.format(name, dataset_parent_dir_path))

    df = pd.read_fwf(dataset_path)
        
    data = df.values
    labels = data[:,0]
    samples = data[:,1:]
    samples = samples.reshape(samples.shape[0], 1, samples.shape[1])

    labels[labels == -1] = 0

    samples, norm_params = normalize_samples(samples, normalization, norm_params)
    
    return samples, labels, norm_params




def load_ecg_heartbeat_categorization_dataset(name, is_testset=False, normalization = None, norm_params = None):
    samples = None
    labels = None

    dataset_dir = '/home/isimic/projects/tsc/data/AdditionalUnivariateTimeSeriesDatasets/ECG Heartbeat Categorization Dataset'
    if is_testset:
        suffix = '_test'
    else:
        suffix = '_train'
    data = pd.read_csv(os.path.join(dataset_dir, 'mitbih_train.csv'), header=None).values
    samples, labels = data[:,:-1], data[:,-1]

    samples = samples.reshape(samples.shape[0], 1, -1)
    samples, norm_params = normalize_samples(samples, normalization, norm_params)

    return samples, labels, norm_params


def load_heartbeat_sound_dataset(is_testset=False, normalization = None, norm_params = None):
    

    label_ids = {
        'normal'   : 0.,
        'artifact' : 1.,
        'murmur'   : 2.,
        'extrahls' : 3.
    }

    set_name = 'set_a'
    parent_dir = '/home/isimic/projects/tsc/data/AdditionalUnivariateTimeSeriesDatasets/Heartbeat Sounds/'

    # Get labels
    labels_dir = os.path.join(parent_dir, '{}.csv'.format(set_name))
    labels_df = pd.read_csv(labels_dir)
    labels_df = labels_df[labels_df['label'].notna()]

    for l in label_ids:
        labels_df.replace({'label' : l}, label_ids[l], inplace=True)

    labels = np.array(labels_df['label'])

    # get samples
    sample_list = []
    max_length = 0
    for index, row in labels_df.iterrows():
        sample_path = os.path.join(parent_dir, row.fname)
        sample = np.array(read(sample_path)[1])
        # samples are int16, so min is -32768 and max is +32767
        min_val = -32768
        max_val = 32767
        sample = (sample - min_val) / (max_val - min_val)
        if len(sample) > max_length:
            max_length = len(sample)
        sample_list.append(sample)
        
    samples = np.zeros((len(sample_list), 1, max_length))
    for i, s in enumerate(sample_list):
        samples[i,0,:s.shape[0]] = s

    samples, norm_params = normalize_samples(samples, normalization, norm_params)

    return samples, labels, norm_params
        
    

def load_multivariate_UCR_dataset(name, is_testset=False, normalization = None, norm_params = None):
    '''Loads the raw data of one of the UCR multivariate datasets

    Args:
        name (str): directory name of the dataset
        is_testset (bool): if set to True, the testset data will be loaded instead of the training set (default False)

    Returns:
        samples (list)
        labels (list)

    Raises:
        FileNotFoundError: If the dataset that has to be loaded does not exist

    Datasets downloaded from: http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_arff.zip
    '''

    samples = None
    labels = None

    dataset_parent_dir_path = 'data/UCR_datasets/Multivariate'
    extension = '.arff'
    if is_testset:
        suffix = '_TEST'
    else:
        suffix = '_TRAIN'

    dataset_path = os.path.join(dataset_parent_dir_path, name, name + suffix + extension)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError('A dataset with the name "{}" does not exist under the path "{}"'.format(name, dataset_parent_dir_path))

    data, meta = arff.loadarff(dataset_path)
    # Examples of structure of data
    # data[0] <- sample 1, data + labels
    # data[0][1] <- sample 1, label
    # data[0][0][0] <- sample 1, data of first channel
    # data[0][0][1] <- sample 1, data of second channel

    root = np.array(data[:].tolist())

    data = root[:,0]
    data_channels = np.array(data[:].tolist())
    samples = np.array(data_channels[:].tolist())

    labels = root[:,1]
    try:
        float(labels[0])
        labels = labels.astype('float')
    except:
        labels = labels.astype('str')
    
    return samples, labels


def store_relevance_heatmap(path, sample, relevance, method_name):
    if c.store_plots:
        if c.store_wide_figures:
            plt.figure(figsize=c.wide_figure_dimensions)

        sample_1d = sample.squeeze()

        x_min = -0.5; x_max = len(sample_1d) - 0.5
        y_min = sample_1d.min(); y_max = sample_1d.max()
        y_min += y_min * 0.1 # add bottom margin
        y_max += y_max * 0.1 # add top margin

        relevance_img = relevance.reshape(1, -1)
        relevance_img = np.tile(relevance, (100,1) )

        plt.title('Relevance as heatmap using {}'.format(method_name), fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.plot(sample_1d, color='blue', linewidth=2, marker='o')
        plt.imshow(relevance_img, extent=[x_min, x_max, y_min, y_max], cmap='YlOrRd', interpolation='nearest', aspect='auto')
        plt.savefig(os.path.join(path, 'relevance_heatmap_{}.png'.format(method_name)))
        plt.close()



def store_relevance_plot(path, relevance, method_name):
    if c.store_plots:
        if c.store_wide_figures:
            plt.figure(figsize=c.wide_figure_dimensions)
        plt.title('Input relevance using {}'.format(method_name))
        plt.plot(relevance, color='blue')
        plt.savefig(os.path.join(path, 'relevance_{}.png'.format(method_name)))
        plt.close()



def store_prediction_change(path, prediction_change, method_name, permutation_method, auc_value):
    if c.store_plots:
        if c.store_wide_figures:
            plt.figure(figsize=c.wide_figure_dimensions)
        plt.title('Prediction change using {}; AUC = {:.2f}'.format(method_name, auc_value))
        plt.ylim(-5, 105)
        plt.plot(prediction_change, color='blue')
        plt.savefig(os.path.join(path, 'prediction_change_{}_{}.png'.format(method_name, permutation_method)))
        plt.close()

    # Store prediction change as json
    pred_change = {'prediction_change' : prediction_change.tolist()}
    with open(os.path.join(path, 'prediction_change_{}_{}.json'.format(method_name, permutation_method)), 'w') as outfile:
        json.dump(pred_change, outfile, indent=2)



def perturb_using_list(target_sample, model, target_output, input_relevance, values = []):
    predictions = []

    sample = np.array(target_sample, copy=True)
    x_batch = torch.tensor(sample,dtype=torch.float32).unsqueeze(0)
    if c.is_gpu_available:
        x_batch = x_batch.cuda()

    output = torch.exp(model(x_batch)).squeeze()
    initial_prediction_value = round(output[target_output].item() * 10000) / 100
    predictions.append(initial_prediction_value)

    indexes = input_relevance.argsort()[::-1]
    modified_timesteps = []

    for i in indexes:
        sample[0][i] = values[0][i]
        modified_timesteps.append(i)
        x_batch = torch.tensor(sample,dtype=torch.float32).unsqueeze(0)
        if c.is_gpu_available:
            x_batch = x_batch.cuda()

        output = torch.exp(model(x_batch)).squeeze()
        prediction_value = round(output[target_output].item() * 10000) / 100
        predictions.append(prediction_value)
        if prediction_value < c.perturbation_threshold:
            break

    predictions = np.array(predictions)

    return predictions, modified_timesteps



def perturb_using_value(target_sample, model, target_output, input_relevance, value):
    value_list = np.ones(target_sample.shape)
    value_list *= value
    predictions, modified_timesteps = perturb_using_list(target_sample, model, target_output, input_relevance, value_list)

    return predictions, modified_timesteps


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def padded_moving_average(a, n=3) :
    padded = np.pad(a, n // 2, 'constant', constant_values=(a[0], a[-1]))
    return moving_average(padded, n)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def profile(fnc):
    
    """A decorator that uses cProfile to profile a function"""
    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        ret_val = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats(10)
        print(s.getvalue())
        return ret_val

    return inner