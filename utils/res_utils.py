import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

def degradation_score(morf_values, lerf_values):
    '''Computes the normalized degradation score (average difference between morf and lerf perturbation curves)

    Parameters:
        morf_values (list): list of model predictions after perturbing the input in the MoRF order
        lerf_values (list): list of model predictions after perturbing the input in the LeRF order

    Returns:
        The degradation score (float) 
    '''

    pc_morf = np.array(morf_values)
    pc_lerf = np.array(lerf_values)

    ds = np.mean(pc_lerf - pc_morf)

    max_diffed = np.zeros_like(pc_morf)
    max_diffed[1:] = 100 # the maximum theoretical difference at each point is 100, except the first point, since there was no perturbation performed at this point for both curves
    ds_max = np.mean(max_diffed)

    normalized_dds = ds / ds_max # divide the ds by the ds_max to get it into the [-1,1] range

    return normalized_dds

def compute_dataset_dds(morf_pcs, lerf_pcs):
    '''Computes the decaying degradation score of the whole dataset

    Parameters:
        morf_pcs (2D list): list of model predictions after perturbing the input in the MoRF order
        lerf_pcs (2D list): list of model predictions after perturbing the input in the LeRF order

    Returns:
        A list of decaying degradation score values (float list) 
    '''

    dds_values = []
    assert (len(morf_pcs) == len(lerf_pcs)), 'Number of MoRF and LeRF perturbation curves has to be equal'
    for i in range(len(morf_pcs)):
        dds_values.append(decaying_degradation_score(morf_pcs[i], lerf_pcs[i]))

    dds_values = np.array(dds_values)

    return dds_values

def decaying_degradation_score(morf_values, lerf_values):
    '''Computes the normalized decaying degradation score (average weighted difference between morf and lerf perturbation curves)

    Parameters:
        morf_values (list): list of model predictions after perturbing the input in the MoRF order
        lerf_values (list): list of model predictions after perturbing the input in the LeRF order

    Returns:
        The decaying degradation score (float) 
    '''

    pc_morf = np.array(morf_values)
    pc_lerf = np.array(lerf_values)

    diffed = pc_lerf - pc_morf

    linear_weights = np.arange(len(diffed),0,-1) / len(diffed)
    cubic_weights = linear_weights**3

    dds = np.average(diffed, weights=cubic_weights)

    max_diffed = np.zeros_like(diffed)
    max_diffed[1:] = 100 # the maximum theoretical difference at each point is 100, except the first point, since there was no perturbation performed at this point for both curves
    dds_max = np.average(max_diffed, weights=cubic_weights) # compute the highest possible dds value (dependent on pc length)

    normalized_dds = dds / dds_max # divide the dds by the dds_max to get it into the [-1,1] range

    return normalized_dds


def pes(dds_vals):
    '''Computes the perturbation effect size using the decaying degradation score values
    Uses Kerby's simple difference formula
    Source: https://journals.sagepub.com/doi/pdf/10.2466/11.IT.3.1
    Difference between ratio of Favourable pairs and Unfavourable r = f - u

    Parameters:
        dds_vals (list): list of decaying degradation scores of each sample

    Returns:
        The perturbation effect size (float) 
    '''
    if isinstance(dds_vals, list):
        dds_vals = np.array(dds_vals)

    f = len(dds_vals[dds_vals > 0]) / len(dds_vals)
    u = len(dds_vals[dds_vals < 0]) / len(dds_vals)
    pes_dds = f - u
    return pes_dds

def rank_biserial(series_1, series_2):
    f_counter = 0
    u_counter = 0
    
    for i in range(len(series_1)):
        if series_1[i] < series_2[i]:
            f_counter += 1
        if series_2[i] < series_1[i]:
            u_counter += 1
    
    f = f_counter / len(series_1)
    u = u_counter / len(series_1)
    
    return f - u


def combined_mean(means, num_elements=None):
    """
    Calculates the combined mean of multiple samples.

    Args:
        means (list): A list of means of the samples.
        num_elements (list): A list of the number of elements in each sample. If None, the same number of samples is expected for all means

    Returns:
        float: The combined mean of the samples.
    """
    if len(means) == 1:
        return means[0]
    
    if num_elements is None:
        num_elements = np.ones_like(means)

    total_sum = 0
    total_num_elements = 0
    for i in range(len(means)):
        total_sum += means[i] * num_elements[i]
        total_num_elements += num_elements[i]
    return total_sum / total_num_elements



def combined_stddev(stddevs, num_elements):
    """
    Calculates the combined standard deviation of multiple samples.

    Args:
        stddevs (list): A list of standard deviations of the samples.
        num_elements (list): A list of the number of elements in each sample. If None, the same number of samples is expected for all means

    Returns:
        float: The combined standard deviation of the samples.
    """
    if len(stddevs) == 1:
        return stddevs[0]

    if len(num_elements) != len(stddevs):
        raise 'len(num_elements) has to be equal to len(stddevs)'
        
    # Calculate the total number of elements
    total_num_elements = sum(num_elements)
    
    # Calculate the weighted average of the standard deviations
    weighted_stddevs = []
    for i in range(len(stddevs)):
        weight = num_elements[i] / total_num_elements
        weighted_stddevs.append(stddevs[i] * math.sqrt(weight))
    weighted_avg_stddev = math.sqrt(sum([stddev ** 2 for stddev in weighted_stddevs]))
    
    # Calculate the combined standard deviation
    combined_stddev = weighted_avg_stddev * math.sqrt(total_num_elements / (total_num_elements - 1))
    
    return combined_stddev


def CMI(dds, pes):
    """
    Calculates the consistency-magnitude-index (CMI), which is the harmonic mean of PES and DDS if they have the same sign. Otherwise 0

    Args:
        dds (float): decaying degradation score - [-1,1]
        pes (float): perturbation effect size - [-1,1]

    Returns:
        float: consistency-magnitude-index (CMI) [0,1]
    """
    if pes * dds <= 0:
        return 0
    else:
        return 2 / ( (1 / abs(dds)) + (1 / abs(pes)) )


def get_row_id(a, b):
    return '{} - {}'.format(a, b)


def create_heatmap(df, type, plt_title, figsize, hlines=[], vlines=[], alpha=0.05):
    sns.set(rc={'figure.figsize':figsize})
    plt.title(plt_title, fontsize = 18)
    if type == 'nh1' or type == 'nh2':
        heatmap = sns.heatmap(df, annot=False, cmap='coolwarm',linewidths=.1, square=True, vmin=alpha, vmax=alpha)
    elif type == 'es':
        heatmap = sns.heatmap(df, annot=False, cmap='coolwarm',linewidths=.1, square=True, vmin=-1, vmax=1)
    elif type == 'std_es':
        heatmap = sns.heatmap(df, annot=False, cmap='Reds',linewidths=.1, square=True, vmin=0, vmax=0.4)
    elif type == 'aad':
        heatmap = sns.heatmap(df, annot=False, cmap='RdYlGn',linewidths=.1, square=True, vmin=-100, vmax=100)
        # heatmap = sns.heatmap(df, annot=False, cmap='Greens',linewidths=.1, square=True, vmin=-100, vmax=100)
    elif type == 'aupc':
        heatmap = sns.heatmap(df, annot=False, cmap='Reds',linewidths=.1, square=True, vmin=0, vmax=100)
    elif type == 'std_aad':
        heatmap = sns.heatmap(df, annot=False, cmap='Reds',linewidths=.1, square=True, vmin=0, vmax=25)
    elif type == 'count_neg':
        heatmap = sns.heatmap(df, annot=False, cmap='Reds',linewidths=.1, square=True, vmin=0, vmax=5)
    else:
        heatmap = sns.heatmap(df, annot=False, cmap='coolwarm',linewidths=.1, square=True)

    return heatmap


def save_results_df(df, res_dir, f_name, plt_title, type, figsize, store_transposed_dfs = False, alpha=0.05):
    # Remove 'Captum' from the attribution method names
    new_col_names = {}
    for col_name in df.columns.to_list():
        if 'Captum' in col_name:
            new_col_names[col_name] = col_name.replace('Captum', '')
    df = df.rename(columns=new_col_names)

    rss = list(set([x.split(' - ')[0] for x in df.index]))
    pms = list(set([x.split(' - ')[1] for x in df.index]))
    ams = df.columns

    heatmap = create_heatmap(df, type, plt_title, figsize, alpha=alpha)

    hlines = np.arange(0,df.shape[0],len(pms))[1:].tolist()
    heatmap.hlines(hlines, *heatmap.get_xlim(), colors='black', linewidth=2)

    if 'null_hypothesis_2' in f_name:
        vlines = np.arange(0,df.shape[0],len(pms))[1:].tolist()
        heatmap.vlines(vlines, *heatmap.get_ylim(), colors='black', linewidth=2)

    # # y-axis: windows size + perturbation method; x-axis: attribution method
    fig = heatmap.get_figure()
    fig.savefig(os.path.join(res_dir, '{}_rs_pm.png'.format(f_name)))
    plt.close('all')
    df.to_csv(os.path.join(res_dir, '{}_rs_pm.csv'.format(f_name)))

    if store_transposed_dfs:
        regrouped_df = df.copy(deep=True)
        regrouped_df['window-size'] = [x.split(' - ')[0] for x in regrouped_df.index]
        regrouped_df['perturbation method'] = [x.split(' - ')[1] for x in regrouped_df.index]
        regrouped_df = regrouped_df.reset_index(drop=True)
        cols = regrouped_df.columns.to_list()
        cols.remove('perturbation method')
        cols.remove('window-size')

        # y-axis: window size + attribution method; x-axis: perturbation method
        rs_am_df = regrouped_df.pivot(index='perturbation method', columns = 'window-size', values=cols)
        rs_am_df.columns = rs_am_df.columns.to_flat_index()
        rs_am_df = rs_am_df.transpose()
        new_index = ['{} - {}'.format(item[1], item[0])   for item in rs_am_df.index.to_list()]
        rs_am_df.index = new_index
        rs_am_df.columns.name = ''
        rs_am_df.sort_index(inplace=True)
        

        new_figsize = (15, 20)
        heatmap = create_heatmap(rs_am_df, type, plt_title, new_figsize)

        hlines = np.arange(0,rs_am_df.shape[0],len(ams))[1:].tolist()
        heatmap.hlines(hlines, *heatmap.get_xlim(), colors='black', linewidth=2)

        fig = heatmap.get_figure()
        fig.savefig(os.path.join(res_dir, '{}_rs_am.png'.format(f_name)))
        plt.close('all')
        rs_am_df.to_csv(os.path.join(res_dir, '{}_rs_am.csv'.format(f_name)))

        # y-axis: attribution method + perturbation method; x-axis: region size
        am_pm_df = regrouped_df.pivot(index='window-size', columns = 'perturbation method', values=cols)
        am_pm_df.columns = am_pm_df.columns.to_flat_index()
        am_pm_df = am_pm_df.transpose()
        new_index = ['{} - {}'.format(item[0], item[1])   for item in am_pm_df.index.to_list()]
        am_pm_df.index = new_index
        am_pm_df = am_pm_df.sort_index()

        am_pm_df.columns.name = ''
        cols = am_pm_df.columns.to_list()
        cols.sort(key = lambda x: int(x))
        am_pm_df = am_pm_df.reindex(columns=cols)

        heatmap = create_heatmap(am_pm_df, type, plt_title, figsize)
        hlines = np.arange(0,am_pm_df.shape[0],len(pms))[1:].tolist()
        heatmap.hlines(hlines, *heatmap.get_xlim(), colors='black', linewidth=2)
        fig = heatmap.get_figure()
        fig.savefig(os.path.join(res_dir, '{}_am_pm.png'.format(f_name)))
        plt.close('all')
        am_pm_df.to_csv(os.path.join(res_dir, '{}_am_pm.csv'.format(f_name)))


def plot_barh_on_axis(ax, x_vals, y_vals, title, xmin=-1.05, xmax=1.05):
    colors = []
    for x_val in x_vals:
        if x_val == x_vals.max():
            colors.append('darkgreen')
        else:
            colors.append('grey')

    ax.barh(y_vals, x_vals, color=colors)
    ax.set_title(title, fontsize=16)
    ax.set_xlim(xmin,xmax)