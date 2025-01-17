import os
import pymongo
from configs.dbconfig import dbConfig
from identify_zero_class import identify_zero_class
import numpy as np
import pandas as pd
from datetime import datetime
import multiprocessing as mp
from utils.res_utils import rank_biserial
from utils.res_utils import pes, degradation_score, decaying_degradation_score
from tqdm import tqdm
from scipy.stats import shapiro

DISABLE_TQDM = False

def iqr(x):
    q75, q25 = np.percentile(x, [75 ,25])
    iqr = q75 - q25
    return iqr

def normality_from_pvalue(pval):
    if pval < 0.05:
        return 'NO'
    else:
        return 'YES'

def add_results_metrics(res_file_obj, metrics_obj, metric):
    values = np.array(metrics_obj[metric])

    if '#Samples' not in res_file_obj:
        res_file_obj['#Samples'] = len(values)

    shapiro_test = shapiro(values)

    res_file_obj['{} - {}'.format(metric, 'mean')]   = values.mean()
    res_file_obj['{} - {}'.format(metric, 'std')]    = values.std()

    res_file_obj['{} - {}'.format(metric, 'median')] = np.median(values)

    # res_file_obj['{} - {}'.format(metric, 'iqr')]    = iqr(values)
    q75, q25 = np.percentile(values, [75 ,25])
    res_file_obj['{} - {}'.format(metric, 'quartile-25')] = q25
    res_file_obj['{} - {}'.format(metric, 'quartile-75')] = q75
    res_file_obj['{} - {}'.format(metric, 'iqr')] = q75 - q25

    res_file_obj['{} - {}'.format(metric, 'Shapiro-Wilk test [p-value]')]    = shapiro_test.pvalue
    res_file_obj['{} - {}'.format(metric, 'Shapiro-Wilk test [statistic]')]    = shapiro_test.statistic


def perform_sanity_check():
    # For a dataset, all models, and attribution methods have to have the same number of samples
    dbclient = pymongo.MongoClient(dbConfig.url)
    db = dbclient[dbConfig.dbName]
    results_collection = db[dbConfig.results_collection]

    datasets = results_collection.distinct('dataset name')

    for dataset in datasets:
        model_architectures = results_collection.find({'dataset name': dataset }).distinct('model architecture')

        samples_per_fam = []
        for model_arch in model_architectures:
            seeds = results_collection.find({
                'dataset name': dataset,
                'model architecture' : model_arch
            }).distinct('model seed')
            for seed in seeds:
                samples_x_fam = results_collection.count_documents({
                    'dataset name': dataset,
                    'model architecture' : model_arch,
                    'model seed' : seed
                })
                n_fams = len(results_collection.find({
                    'dataset name': dataset,
                    'model architecture' : model_arch,
                    'model seed' : seed
                }).distinct('feature attribution method'))

                samples_per_fam.append(samples_x_fam / n_fams)

        assert all(s == samples_per_fam[0] for s in samples_per_fam)


def calculate_quality_metrics(results_path):
    dbclient = pymongo.MongoClient(dbConfig.url)
    db = dbclient[dbConfig.dbName]
    results_collection = db[dbConfig.results_collection]
    datasets             = results_collection.distinct('dataset name')
    results = []
    for dataset in tqdm(datasets, desc='{: <20}'.format('Datasets'), disable=DISABLE_TQDM, position=0):
        model_architectures = results_collection.find({'dataset name': dataset }).distinct('model architecture')
        for model_architecture in tqdm(model_architectures, desc='{: <20}'.format('Model Architectures'), disable=DISABLE_TQDM, position=1, leave=False):
            seeds = results_collection.find({
                'dataset name' : dataset,
                'model architecture' : model_architecture
            }).distinct('model seed')
            for seed_num, seed in enumerate(tqdm(seeds, desc='{: <20}'.format('Seeds'), disable=DISABLE_TQDM, position=2, leave=False)):
                attribution_methods = results_collection.find({
                    'dataset name': dataset,
                    'model architecture' : model_architecture,
                    'model seed' : seed
                }).distinct('feature attribution method')

                for am in tqdm(attribution_methods, desc='{: <20}'.format('Attribution Methods'), disable=DISABLE_TQDM, position=3, leave=False):
                    if DISABLE_TQDM:
                        print('Calculate faithfulness metrics: {} | {} ({}/{}) | {}'.format(dataset, model_architecture, seed_num + 1, len(seeds), am))
                    filter_obj = {
                        'dataset name' : dataset,
                        'model architecture' : model_architecture,
                        'model seed' : seed,
                        'feature attribution method' : am
                    }

                    zero_class = identify_zero_class(dataset, model_architecture, seed)

                    results_subset = results_collection.find(filter_obj)

                    # Aggregate all MoRF and LeRF AUPCs
                    quality_metrics = {}
                    for sample in results_subset:

                        for sample_res in sample['results']:
                            # To get the area, we need to sum up all the model predictions for each perturbation step
                            # However, since we want a normalize the value by perturbation steps, we calculate the mean instead
                            pc_morf = np.array(sample_res['perturbation results']['MoRF'])
                            pc_lerf = np.array(sample_res['perturbation results']['LeRF'])

                            aupc_morf = pc_morf.mean()#.round(decimals=2)
                            aupc_lerf = pc_lerf.mean()#.round(decimals=2)

                            ds_val = degradation_score(pc_morf, pc_lerf)
                            dds_val = decaying_degradation_score(pc_morf, pc_lerf)

                            key = (sample_res['perturbation method'], sample_res['region size']) 
                            if key not in quality_metrics:
                                quality_metrics[key] = {}

                            class_name = str(sample['class id'])

                            if class_name not in quality_metrics[key]:
                                quality_metrics[key][class_name] = {
                                    'AUPC MoRF' : [],
                                    'AUPC LeRF' : [],
                                    'DS' : [],
                                    'DDS' : []
                                }

                            quality_metrics[key][class_name]['AUPC MoRF'].append(aupc_morf)
                            quality_metrics[key][class_name]['AUPC LeRF'].append(aupc_lerf)
                            quality_metrics[key][class_name]['DS'].append(ds_val)
                            quality_metrics[key][class_name]['DDS'].append(dds_val)

                    # compute experiment results
                    for key in quality_metrics:
                        perturbation_method = key[0]
                        region_size = key[1]

                        all_classes = {
                            'AUPC MoRF' : [],
                            'AUPC LeRF' : [],
                            'DS' : [],
                            'DDS' : [],
                        }

                        all_classes_no_zero_class = {
                            'AUPC MoRF' : [],
                            'AUPC LeRF' : [],
                            'DS' : [],
                            'DDS' : [],
                        }

                        ########################################################################################
                        # Store results of individual classes
                        for class_name in quality_metrics[key]:
                            res_file_obj = {
                                'Dataset' : filter_obj['dataset name'],
                                'Model' : filter_obj['model architecture'],
                                'Seed' : filter_obj['model seed'],
                                'Attribution Method' : filter_obj['feature attribution method'],
                                'Perturbation Method' : perturbation_method,
                                'Region Size' : region_size,
                                'Class name' : class_name,
                                'is_zero_class' : (class_name == str(zero_class))
                            }
                            add_results_metrics(res_file_obj, quality_metrics[key][class_name], 'AUPC MoRF')
                            add_results_metrics(res_file_obj, quality_metrics[key][class_name], 'AUPC LeRF')
                            add_results_metrics(res_file_obj, quality_metrics[key][class_name], 'DS')
                            add_results_metrics(res_file_obj, quality_metrics[key][class_name], 'DDS')

                            all_classes['AUPC MoRF'] += quality_metrics[key][class_name]['AUPC MoRF']
                            all_classes['AUPC LeRF'] += quality_metrics[key][class_name]['AUPC LeRF']
                            all_classes['DS'] += quality_metrics[key][class_name]['DS']
                            all_classes['DDS'] += quality_metrics[key][class_name]['DDS']

                            if class_name != str(zero_class):
                                all_classes_no_zero_class['AUPC MoRF'] += quality_metrics[key][class_name]['AUPC MoRF']
                                all_classes_no_zero_class['AUPC LeRF'] += quality_metrics[key][class_name]['AUPC LeRF']
                                all_classes_no_zero_class['DS'] += quality_metrics[key][class_name]['DS']
                                all_classes_no_zero_class['DDS'] += quality_metrics[key][class_name]['DDS']

                            # # Compute and add effect size based on the AUPC MoRF and AUPC LeRF fields
                            res_file_obj['PES - old'] = rank_biserial(quality_metrics[key][class_name]['AUPC MoRF'], quality_metrics[key][class_name]['AUPC LeRF'])
                            # Compute and add effect size based on DDS
                            res_file_obj['PES'] = pes(quality_metrics[key][class_name]['DDS'])
                            
                            results.append(res_file_obj)

                        ########################################################################################
                        # Store results over all samples of all classes (including zero class)
                        res_file_obj = {
                            'Dataset' : filter_obj['dataset name'],
                            'Model' : filter_obj['model architecture'],
                            'Seed' : filter_obj['model seed'],
                            'Attribution Method' : filter_obj['feature attribution method'],
                            'Perturbation Method' : perturbation_method,
                            'Region Size' : region_size,
                            'Class name' : 'all classes',
                            'is_zero_class' : False
                        }
                        add_results_metrics(res_file_obj, all_classes, 'AUPC MoRF')
                        add_results_metrics(res_file_obj, all_classes, 'AUPC LeRF')
                        add_results_metrics(res_file_obj, all_classes, 'DS')
                        add_results_metrics(res_file_obj, all_classes, 'DDS')
                        # # Compute and add effect size based on the AUPC MoRF and AUPC LeRF fields
                        res_file_obj['PES - old'] = rank_biserial(all_classes['AUPC MoRF'], all_classes['AUPC LeRF'])
                        # Compute and add effect size based on DDS
                        res_file_obj['PES'] = pes(all_classes['DDS'])
                        
                        results.append(res_file_obj)

                        ########################################################################################
                        # Store results over all samples of all classes (excluding zero class)
                        res_file_obj = {
                            'Dataset' : filter_obj['dataset name'],
                            'Model' : filter_obj['model architecture'],
                            'Seed' : filter_obj['model seed'],
                            'Attribution Method' : filter_obj['feature attribution method'],
                            'Perturbation Method' : perturbation_method,
                            'Region Size' : region_size,
                            'Class name' : 'all classes - no zero class',
                            'is_zero_class' : False
                        }
                        add_results_metrics(res_file_obj, all_classes_no_zero_class, 'AUPC MoRF')
                        add_results_metrics(res_file_obj, all_classes_no_zero_class, 'AUPC LeRF')
                        add_results_metrics(res_file_obj, all_classes_no_zero_class, 'DS')
                        add_results_metrics(res_file_obj, all_classes_no_zero_class, 'DDS')
                        # # Compute and add effect size based on the AUPC MoRF and AUPC LeRF fields
                        res_file_obj['PES - old'] = rank_biserial(all_classes_no_zero_class['AUPC MoRF'], all_classes_no_zero_class['AUPC LeRF'])
                        # Compute and add effect size based on DDS
                        res_file_obj['PES'] = pes(all_classes_no_zero_class['DDS'])
                        
                        results.append(res_file_obj)
                        

    df = pd.DataFrame(results)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    df.to_csv(os.path.join(results_path,'quality metrics.csv'))    


if __name__ == "__main__":
    start_time = datetime.now()

    now = datetime.now()
    timestamp = now.strftime('%Y_%m_%d__%H_%M_%S')    
    
    results_path = 'quality_metrics' # directory where the computed faithfulness metrics will be stored
    
    perform_sanity_check()

    calculate_quality_metrics(results_path)

    end_time = datetime.now()
    total_time = end_time - start_time
    print("**** Total time for calculations took: {} day(s) and {:02d}:{:02d}:{:02d}".format(total_time.days, (total_time.seconds // 3600) % 24, (total_time.seconds // 60) % 60, total_time.seconds % 60))