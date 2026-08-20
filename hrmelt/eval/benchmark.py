"""
This file will evaluate the predictions of all models and create 
a benchmark table of the performance metrics across models. This
benchmark assumes that model predictions are already computed,
e.g., via predict.py, on all images in the validation set. The 
model predictions should be stored in a folder, e.g., 
'runs/unet/data_v1_4/predictions/*'. The benchmark is parallelized 
using torch.

Author: Björn Lütjens
"""
import argparse
import os
from typing import Sequence

import numpy as np
from tqdm import tqdm
from pprint import pprint
from pathlib import Path
from osgeo import gdal # rasterio in dataloader uses
    # gdal. Need to import gdal to suppress warning msg. 
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from datetime import datetime
import pandas as pd
from functools import partial

import hrmelt.eval.metrics as metrics
from hrmelt.dataset import HRMeltDataset
from hrmelt.dataset import get_filepaths_from_csv
from hrmelt.utils.utils import get_size_of_tif
from hrmelt.utils.utils import MaskedLoss
from hrmelt.utils.utils import set_num_workers
from hrmelt.utils.utils import lookup_torch_dtype
from hrmelt.utils.utils import _worker_init_fn
from hrmelt.eval.metrics import MaskedSSIM
from hrmelt.eval.metrics import MaskedPrecision
from hrmelt.eval.metrics import MaskedRecall
from hrmelt.eval.metrics import MaskedR2
from hrmelt.eval.metrics import CountValidPx

from tqdm import tqdm

class HRMeltDatasetPredictions(HRMeltDataset):
    def __init__(self, cfg, split='val', verbose=False, 
                 model_keys=['unet'],
                 paths_predictions=['./runs/unet/data_v1_4/predictions/'],
                 sort=True):
        '''
            Child class of HRMeltDataset that adds the paths of model 
            predictions to the data list. With this the __getitem__() function
            can return the corresponding prediction ground-truth image under 'input'
            a mask of invalid pixels under 'targets_mask', and the ground-truth 
            image in 'targets'. The dataset is configured to only return images 
            at the size of the full-scale tif. Otherwise uses the same __getitem__()
            fn as the super class.

        Args:
            cfg, split, verbose: see parent class
            model_key [str]: List of keys to model that was used to generate the predictions
            paths_predictions [str]: List of paths to folder with all precalculated
             predictions of the model
            sort bool: If True, the returned data will be sorted by timestamp
        '''
        # Call the parent class constructor, adding the model predictions and
        #  static landmask to the returned 'inputs'.
        cfg['in_keys'] = ['predictions_' + model_key for model_key in model_keys]
        super().__init__(cfg=cfg, split=split, verbose=verbose)

        self.model_keys = model_keys
        # Initialize img_size to the size of the full image, by loading 
        #  a sample image and getting its size
        if 'path_melt_reference' in cfg:
            self.tif_size = get_size_of_tif(str(Path(cfg['data_root'])/Path(cfg['path_melt_reference'])))
        else:
            try:
                sample_idx = 0 # Index to the first sample of the dataset
                sample_key = list(self.data[sample_idx].keys())[0]
                sample_path = self.data[sample_idx][sample_key]
                self.tif_size = get_size_of_tif(str(sample_path))
            except:
                print(f'Could not reference .tif at {sample_path} to read size of study area. Please'\
                      f'set the config argument cfg["path_melt_reference"] = /path/to/sample.tif')
        cfg['img_size'] = self.tif_size # e.g., (2863, 1633)

        # Concatenate keys of all in- and output channels
        data_keys = self.cfg['in_keys'] + ['melt']
        if self.split == 'deploy':
            data_keys.remove('melt')

        # Overwrite data dictionary with dictionary that contains filepaths 
        #  of predictions
        paths_data_keys = {}
        if self.split != 'deploy':
            paths_data_keys['path_melt'] = Path(cfg['data_root'])/Path(self.cfg['path_melt'])

        for model_key, path_predictions in zip(model_keys, paths_predictions):
            paths_data_keys[f'path_predictions_{model_key}'] = Path(path_predictions)
        # 'path_landmask': Path(cfg['data_root'])/Path(self.cfg['path_landmask']),

        self.data, self.filenames = get_filepaths_from_csv(
            path_csv=self.cfg[f'path_{split}_split_csv'],
            data_root='', # we set data_root empty because, we already incorporated it in paths_data_keys
            data_keys=data_keys,
            subpaths=paths_data_keys,
            sort=sort)

        if verbose:
            print('Loading data from:')
            pprint(self.data[0])
    
    def __getitem__(self, idx):
        """
        Args:
            idx Index into all tiles in dataset
        """
        return super().__getitem__(idx)

def benchmark_metrics(metrics_fn, model_keys, device, cfg, split='val', return_filenames=False, verbose=None):
    """
    Computes all metrics in metrics_fn on the predictions of every
     model in model_keys and prints the results to console. Assumes
     that the predictions are already computed and stored in a folder

    Args:
        metrics_fn {metric_key1: metric_fn1,
                    ...}: Dictionary with all desired metrics and the 
                    function to compute them
        model_keys [model_key1, ...]: List of models that will be evaluated
        device torch.device: Device to use for computation
        cfg dict: Config dictionary with all necessary paths and parameters
        return_filenames bool: this will save the evaluated filenames to the output dict.
        verbose bool: if verbose is not explicitly set the cfg['verbose'] param will be used
    Returns:
        metrics = {'model_name': 
                      {'MSE': n_val_images * [], 
                      'MSE_mean': [], 
                      'MSE_std': [],
                      'MAE': ...}},
    """
    # if not specified verbosity will be set by config
    verbose = verbose if verbose is not None else cfg['verbose']
    # Init metrics dictionary
    metrics = {}
    for model_key in model_keys:
        metrics[model_key] = {}
        metrics[model_key]['file_names'] = []
        for metrics_key in metrics_fn.keys():
            metrics[model_key][metrics_key] = []

    # set dataloader config
    init_fn_with_cfg = partial(_worker_init_fn, seed=cfg['seed'])
    if verbose:
        pprint(cfg)
    dataloader_args = dict(shuffle=False, 
                    drop_last=False,
                    batch_size=cfg['prediction_batch_size'], 
                    num_workers=cfg['num_workers'],
                    pin_memory=True,
                    worker_init_fn=init_fn_with_cfg,
                    )
    dtype = lookup_torch_dtype(cfg['dtype'])
    cfg['in_keys_static'] = ['landmask']

    # use path_eval when in training evaluation is done.
    path_predictions = [cfg['path_eval']] if 'path_eval' in cfg else cfg['paths_predictions']

    # Iterate over each model:
    for model_key, path_predictions in zip(model_keys, path_predictions):
        print(f'Computing metrics for model: {model_key}')
        # Initialize dataset
        dataset = HRMeltDatasetPredictions(cfg=cfg,
                    split=split,
                    verbose=verbose,
                    model_keys=[model_key],
                    paths_predictions=[path_predictions],)
        # Create data loader
        dataloader = DataLoader(dataset, **dataloader_args)

        # Compute metrics on every batch of images
        for i, batch in tqdm(enumerate(dataloader), disable=not verbose):
            with torch.no_grad():
                inputs, targets, targets_mask, meta = batch

                inputs = inputs.to(device=device, dtype=dtype, memory_format=torch.channels_last)
                targets = targets.to(device=device, dtype=dtype)
                targets_mask = targets_mask.to(device=device, dtype=dtype) # targets_mask includes invalid SAR pixels and landmask
                # Get the model prediction from the inputs
                pred_model_ch_idx = dataset.get_channel_idx(meta, f'predictions_{model_key}')
                pred_model = inputs[:,pred_model_ch_idx:pred_model_ch_idx+1,...]

                # Compute each metric and add to dictionary.
                # The output of the metrics function is converted into a list of len==batch_size.
                for metric_key in metrics_fn.keys():
                    if metric_key == 'MaskedSSIM' and i == 0:
                        print('INFO: computing MaskedSSIM over full image may cause OOM seg fault.'\
                              'Disable using periodical_evaluation_include_ssim.')
                    metric_val = metrics_fn[metric_key](input=pred_model, target=targets, mask=targets_mask)
                    metrics[model_key][metric_key].extend(metric_val.cpu().numpy())

                del pred_model, inputs, targets, targets_mask
            
            if return_filenames:
                # Adding file names
                metrics[model_key]['file_names'].extend([os.path.basename(path) for path in meta['path_melt']])

    # Compute mean and std of metrics
    for model_key in model_keys:
        for metric_key in metrics_fn.keys():
            metrics[model_key][metric_key+'_mean'] = np.mean(metrics[model_key][metric_key])
            metrics[model_key][metric_key+'_std'] = np.std(metrics[model_key][metric_key])

            if 'ValidPx' in metrics_fn.keys():
                metrics[model_key][metric_key+'_weighted_mean'] = np.average(metrics[model_key][metric_key], weights=metrics[model_key]['ValidPx'])
                variance = np.average((metrics[model_key][metric_key]-metrics[model_key][metric_key+'_weighted_mean'])**2,
                                    weights=metrics[model_key]['ValidPx'])
                metrics[model_key][metric_key+'_weighted_std'] = np.sqrt(variance)

    # Print the metrics summaries
    print('Metrics:')
    for model_key in model_keys:
        for metric_key in metrics[model_key]:
            if metric_key.endswith('_mean') or metric_key.endswith('_std'):
                print(f'{model_key}\t {metric_key}:\t{metrics[model_key][metric_key]}')
    
    return metrics

def log_benchmark_metrics_to_wandb(metrics, wandb_run, metrics_fn_keys):
    """
    # Creating a formated dictionary of the benchmark metrics and logs it to wandb
    Args:
        metrics: dict(): see returned object in benchmark_metrics() for format.
        wandb_run wandb.sdk.wandb_run.Run: wandb run object
        metrics_fn_keys ['MaskedMSE', 'MaskedSSIM', ...]: Metric keys

    result_dict: {
            filename_1: {metric1: value, metric2: value}, 
            filename_2: {metric1: value, metric2: value}, 
            metric1_mean: value, 
            metric1_std: value,
            metric2_mean: value,
            ...}
    """
    # Remove 'model_name' dimension from dict.
    metrics = metrics[next(iter(metrics))]

    # add statistical summaries, e.g., mean, std, to the dict
    result_dict = {}
    for key, value in metrics.items():
        if type(value) == float or type(value) == np.float32 or type(value) == np.float64 or type(value) == int:
            result_dict[key] = value

    # add each file as a new key with a dictionary of the associated metrics
    file_names = metrics.get('file_names', [])
    for key, values in metrics.items():
        if key in metrics_fn_keys:
            for file_name, value in zip(file_names, values):
                if file_name not in result_dict:
                    result_dict[file_name] = {}
                result_dict[file_name][key] = value

    # log to wandb
    wandb_run.log(result_dict, commit=False)

    del result_dict

# Plot the predictions for evaluation.
def plot_predictions_vs_targets(model_keys, device, cfg, verbose=False, dpi=300, plot_errors=False, split='val'):
    """
    Plot all predictions vs. targets
        e.g., plot unet, time_interpolate_sar, interpolate-mar, targets, and targets_mask
    # todo: parallelize if its possible to plot multiple images in parallel in matplotlib
    Args:
        model_keys List(str): List of model keys, e.g., ['unet', 'time_interpolate_sar', 'interpolate_mar'], used for labeling
        device torch.device: Device to use for computation
        cfg dict: Config dictionary
        verbose bool: If true, print verbose logs
        dpi int: Dots per inch for the figure
        plot_errors bool: If true, plot the error (predictions - targets) instead of the predictions for each model
    """
    if plot_errors:
        dir_figures = cfg['path_benchmark_figures'] + 'plot_all_model_errors_vs_targets/'
    else:
        dir_figures = cfg['path_benchmark_figures'] + 'plot_all_model_predictions_vs_targets/'
    print('Plotting all model predictions or errors vs targets in ', dir_figures)
    
    # set dataloader config
    cfg['batch_size'] = 1
    pprint(cfg)
    dataloader_args = dict(shuffle=False, drop_last=False,
                    batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                    pin_memory=True)
    dtype = lookup_torch_dtype(cfg['dtype'])
    # Add landmask to apply landmask before plotting
    cfg['in_keys_static'] = ['landmask']

    # Iterate over each model:
    print(f'Plotting models: {model_keys}')
    # Initialize dataset
    dataset = HRMeltDatasetPredictions(cfg=cfg, 
                split=split, 
                verbose=verbose,
                model_keys=model_keys,
                paths_predictions=cfg['paths_predictions'])
    # Create data loader
    dataloader = DataLoader(dataset, **dataloader_args)

    # Init colormaps
    columns = len(model_keys) + 1
    ticks = columns * [None] # init ticks
    if plot_errors:
        # Errors
        cmaps = len(model_keys) * ['coolwarm'] # init colormaps
        bounds = np.array([-1., -0.9, -0.1, 0.1, 0.9, 1.])
        cnorms = len(model_keys) * [colors.BoundaryNorm(boundaries=bounds, ncolors=256)]
        ticks = len(model_keys) * [bounds.tolist()]
        # setting the tick_labels is a bit of a hack to increase the spacing between numbers
        #  so they don't overlap. alternative would have been to set very small font size or disproportional colorbar
        tick_labels = len(model_keys) * [['-1', '        -.9', '-.1', '.1', '.9        ', '1']]
        # Test out twoslopenorm instead of boundary norm
        cnorms = len(model_keys) * [colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1.)]
        ticks = len(model_keys) * [[None]]
        tick_labels = len(model_keys) * [[None]]

        # Targets, targets_mask, masked_targets
        cmaps.extend(2 * ['viridis'])
        cnorms.extend(2 * [colors.Normalize(vmin=0., vmax=1.)])
        ticks.extend(2 * [[None]])
        tick_labels.extend(2 * [[None]])
    else:
        cmaps = columns * ['viridis'] # init colormaps
        cnorms = columns * [colors.Normalize(vmin=0., vmax=1.)] # init colornorms

    # Compute metrics on every batch of images
    for i, batch in tqdm(enumerate(dataloader)):
        inputs, targets, targets_mask, meta = batch

        inputs = inputs.to(device=device, dtype=dtype, memory_format=torch.channels_last)
        targets = targets.to(device=device, dtype=dtype)
        targets_mask = targets_mask.to(device=device, dtype=dtype) # targets_mask includes invalid SAR pixels and landmask
        
        landmask = inputs[:,0:1,...]
        predictions = inputs[:,1:,...]

        # Calculate error
        if plot_errors:
            predictions = predictions - targets
            # Apply targets_mask to errors; targets_mask is 1 for all invalid pixels
            for i in range(len(model_keys)):
                predictions[:,i:i+1,...] = predictions[:,i:i+1,...] * (1-targets_mask)

        # Apply landmask to predictions and target
        #  Set all values in predictions to zero where landmask is one
        #  We apply landmask instead of targets_mask to illustrate that the model can gap fill into the regions of invalid pixels
        for i in range(len(model_keys)):
            predictions[:,i:i+1,...] = predictions[:,i:i+1,...] * (1-landmask)
        targets = targets * (1-landmask)

        # Init plot
        fig, axs = plt.subplots(1, columns, figsize=(10,4), dpi=dpi)
        # concatenate all images. Ignore landmask on 0th channel.
        imgs = torch.cat((predictions[0], targets[0]),dim=0) # , targets_mask[0]
        titles = [model_key.replace("predictions_", "") for model_key in model_keys] + ['masked_targets'] # 'targets', 'targets_mask']
        for i, img in enumerate(imgs):
            ax = axs[i].imshow(img.cpu().numpy(), cmap=cmaps[i], norm=cnorms[i])
            cbar = plt.colorbar(ax, orientation='horizontal', fraction=0.05, pad=0.01 , spacing='proportional') # ticks=ticks[i]
            if titles[i] == 'masked_targets':
                # Plot mask overlayed onto targets
                axs[i].imshow(targets_mask[0,0].cpu().numpy(), cmap='Set1', interpolation='nearest', alpha=0.8*targets_mask[0,0].cpu().numpy())

            # Set ticks explicitly for error plot, because we want to visualize the midrange predictions between \pm 0.1 to 0.9 error
            if plot_errors:
                cbar.ax.tick_params(labelsize=8)
                if i < len(model_keys):
                    if ticks[i][0] is not None:
                        cbar.ax.set_xticks(ticks[i])
                        cbar.ax.set_xticklabels(tick_labels[i])
            # get title from data keys, but skipping landmask
            axs[i].set_title(titles[i])
            axs[i].axis('off')
            
        timestamp = Path(meta['path_melt'][0]).stem # e.g., 2017_08_23
        if plot_errors:
            fig.suptitle(f'Meltwater fraction error (prediction - target), {timestamp} \n [-1 = underpredict melt, false negative ] [+1 = overpredict melt, false positive]')
        else:
            fig.suptitle(f'Predicted fraction of surface meltwater per 100m grid cell, {timestamp} \n [gray = masked out pixels]')
        plt.tight_layout()

        # Save figure
        Path(dir_figures).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{dir_figures}{timestamp}.png")
        plt.close()

    return 1

# Plot the predictions for evaluation.
def plot_meltwater_extent_over_time(model_keys, device, cfg, split='val', plot_errors=False, model_labels=None):
    """
    Plot the integrated extent of meltwater, np.sum(meltwater, axis=space), over time for each
      model prediction and target. Creates one plot per year in the validation dataset
      with one entry per day.

    Args:
        model_keys List(str): List of model keys, e.g., ['unet', 'time_interpolate_sar', 'interpolate_mar'], used for labeling
        device torch.device: Device to use for computation
        cfg dict: Config dictionary
        plot_errors bool: If true, plot the error (predictions - targets) instead of the predictions for each model
        model_labels dict: If not None, should be a dictionary with 'model_key': 'model_label' used in plot legend
    """
    if plot_errors:
        dir_figures = cfg['path_benchmark_figures'] + 'plot_integrated_meltwater_predictions_over_time/'
    else:
        dir_figures = cfg['path_benchmark_figures'] + 'plot_integrated_meltwater_error_over_time/'
    print('Plotting integrated meltwater over time at ', dir_figures)

    # set dataloader config
    dataloader_args = dict(shuffle=False, drop_last=False,
                    batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                    pin_memory=True)
    # No need for landmask, because targets_mask will contain info on all invalid pixels
    cfg['in_keys_static'] = []

    # Iterate over each model:
    print(f'Plotting models: {model_keys}')
    # Initialize dataset
    dataset = HRMeltDatasetPredictions(cfg=cfg, 
                split=split, 
                model_keys=model_keys, # model_keys will be used as 'in_keys'
                paths_predictions=cfg['paths_predictions'],
                sort=True)
    # Create data loader
    dataloader = DataLoader(dataset, **dataloader_args)

    n_days = len(dataset)
    total_meltwater_preds_per_day = np.zeros((n_days, len(model_keys))) # predictions per model
    total_meltwater_per_day = np.zeros((n_days, 1))
    timestamps_per_day = np.empty((n_days, len(model_keys)), dtype="datetime64[ns]")
    n_valid_pixels_per_day = np.zeros((n_days, 1))

    count = 0 # count number of images processed
    with tqdm(total=len(dataset),unit='img') as pbar:
        for i, batch in enumerate(dataloader):
            inputs, targets, targets_mask, meta = batch
            batch_size = inputs.shape[0] # batch_size might vary for the last batch

            inputs = inputs.to(device=device, dtype=dataset.dtype, memory_format=torch.channels_last)
            targets = targets.to(device=device, dtype=dataset.dtype)
            targets_mask = targets_mask.to(device=device, dtype=dataset.dtype) # targets_mask includes invalid SAR pixels and landmask
            
            predictions = inputs[:,0:,...]

            # Calculate error
            if plot_errors:
                raise NotImplementedError('benchmark.py -> plot_meltwater_extent_over_time() does not support plot_errors yet')
            
            # Get timestamps
            timestamps = [Path(path_melt).stem for path_melt in meta['path_melt']] # e.g., '2017_08_23'
            timestamps = np.array([datetime.strptime(timestamp, "%Y_%m_%d") for timestamp in timestamps]) # Convert from List('YYYY_MM_DD') to np.array(DatetimeIndex)
            timestamps = np.repeat(timestamps[:,None], repeats=len(model_keys), axis=1) # Repeat along model dimension for vectorized plotting

            # Apply targets_mask to predictions and target (1=invalid)
            for i in range(len(model_keys)):
                predictions[:,i:i+1,...] = predictions[:,i:i+1,...] * (1-targets_mask)
            targets = targets * (1-targets_mask)
            n_valid_pixels = torch.sum(1-targets_mask, axis=(-2,-1)) # dims: (batch_size, 1)
            
            # Compute total observed meltwater per day, i.e., sum of all fractional meltwater values that are in valid locations
            total_meltwater = torch.sum(targets, axis=(-2,-1)) # dims: (batch_size, 1)
            total_meltwater_preds = torch.sum(predictions, axis=(-2,-1)) # dims: (batch_size, n_models)
        
            # Store results
            timestamps_per_day[count:count+batch_size] = timestamps
            total_meltwater_preds_per_day[count:count+batch_size] = total_meltwater_preds.cpu().numpy()
            total_meltwater_per_day[count:count+batch_size] = total_meltwater.cpu().numpy()
            n_valid_pixels_per_day[count:count+batch_size] = n_valid_pixels.cpu().numpy()

            count += batch_size
            pbar.update(inputs.shape[0])
            if count == n_days: # for debugging
                break
    
    n_img_pixels = np.asarray(cfg['img_size']).prod()
    valid_observations_per_day = n_valid_pixels_per_day / n_img_pixels # Valid observations per day (as fraction of total image)
    average_meltwater_per_day = total_meltwater_per_day / n_valid_pixels_per_day # Average meltwater fraction per valid pixel

    # Convert to pandas dataframe
    df = pd.DataFrame(data=total_meltwater_preds_per_day[:,:], index=pd.to_datetime(timestamps_per_day[:,0]), columns=model_keys)
    df['total_meltwater_per_day'] = total_meltwater_per_day
    df['n_valid_pixels_per_day'] = n_valid_pixels_per_day
    df['valid_observations_per_day'] = valid_observations_per_day
    df['average_meltwater_per_day'] = average_meltwater_per_day

    for model in model_keys:
        df[f'average_meltwater_per_day_{model}'] = df[model] / df['n_valid_pixels_per_day'] # Average meltwater fraction per valid pixel for each model

    # Assign colors to each model_key
    colors = {'unet': 'tab:blue',
            'time_interpolate_sar': 'tab:olive',
            'interpolate_mar': 'tab:brown',
            'linear_dem': 'tab:orange',
            'deeplabv3': 'tab:red',
            'unet_smp': 'tab:blue',
            'threshold_pmw': 'lightgray',
            'random_forest': 'tab:orange',
            'unet_smp_all': 'tab:blue',
            'unet_smp_no_dem': 'tab:orange',
            'unet_smp_no_mar': 'tab:green',
            'unet_smp_no_pmw': 'tab:red',
            'unet_smp_no_sar': 'tab:pink',
            'unet_smp_only_dem': 'tab:orange',
            'unet_smp_only_mar': 'tab:green',
            'unet_smp_only_pmw': 'tab:red',
            'unet_smp_only_sar': 'tab:pink',
            }
    for model_key in model_keys:
        if model_key not in colors.keys():
            colors[model_key] = 'tab:pink' # plt.cm.Set1.colors[7]

    # Store the ML model comparisons in different folder than the baseline comparisons
    if 'deeplabv3' in model_keys:
        dir_figures = dir_figures + 'deeplabv3/'
    elif 'unet_smp' in model_keys:
        dir_figures = dir_figures + 'unet_smp/'

    # Initialize plot for monthly average climatology
    fig, axs = plt.subplots(1, 1)#, figsize=(10,3), dpi=200)
    axs = [axs]

    # Calculate monthly averages
    df_mon_avg = df.groupby(df.index.month).mean()
    df_mon_avg.index = pd.to_datetime(df_mon_avg.index, format='%m')
    # First, plot targets:
    axs[0].plot(df_mon_avg.index.month_name(), df_mon_avg.average_meltwater_per_day.values, 
                marker='X', linestyle='-', color='black', label='targets')
    # Plot model predictions
    for m, model in enumerate(model_keys):
        axs[0].plot(df_mon_avg.index.month_name(), df_mon_avg[f'average_meltwater_per_day_{model}'].values, 
                    linestyle='--', color=colors[model], label=model_labels[model])
        if m == 0:
            axs[0].set_ylabel('Monthly-averaged meltwater\n fraction per observed pixel', fontsize='large')
            axs[0].set_xlabel('Month', fontsize='large') # Time in YYYY-MM
        axs[0].legend(loc='upper left')
        axs[0].set_ylim((0.,1.))
        plt.tight_layout()
        # Save figure after every added model
        Path(dir_figures).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{dir_figures}avg_meltwater_preds_per_month_{split}_{m}.png")

    plt.close()

    # Initialize plot for daily time-series
    from hrmelt.utils.plotting import split_axes
    import matplotlib.dates as mdates

    # Assign transparencies to each data point, such that point with only a few valid pixels are transparent
    alphas = np.where(df['valid_observations_per_day']<0.2, 0.2, 1.) # (n_days, 1)
    plot_key = 'average_meltwater_per_day'
    years = df.index.year.unique()
    n_years = len(years)
    fig, axs = plt.subplots(1, n_years, sharey=True, facecolor='w',figsize=(15,4),dpi=200)
    # Plot the target data of each year
    for ax, yr in zip(axs, years):
        df_yr = df[df.index.year == yr]
        if plot_key == 'average_meltwater_per_day':
            # Plot meltwater targets
            ax.plot(df_yr.index, df_yr.average_meltwater_per_day, label='targets', color='black', alpha=0.6)
            #[ax.plot(df.index[t], df[f'average_meltwater_per_day'].iloc[t], marker='X', linestyle='None', color='black', alpha=alphas[t]) for t in range(n_days)]
            [ax.plot(df_yr.index[t], df_yr[f'average_meltwater_per_day'].iloc[t], marker='X', linestyle='None', color='black', alpha=alphas[t]) for t in range(len(df_yr))]
            #  Plot predictions as lines over time
            [ax.plot(df_yr.index, df_yr[f'average_meltwater_per_day_{model}'], 
                     linestyle='--', markerfacecolor='None', label=model_labels[model], 
                     color=colors[model], alpha=0.6) for m, model in enumerate(model_keys)]
            #  Then, plot individual data points where data points over just a few valid pixels are more transparent
            [[ax.plot(df_yr.index[t], df_yr[f'average_meltwater_per_day_{model}'].iloc[t], marker='X', markerfacecolor='None', linestyle='None', alpha=alphas[t], color=colors[model]) for t in range(len(df_yr))] for m, model in enumerate(model_keys)]
        else:
            raise NotImplementedError(f'Given plot_key ({plot_key}) is invalid.')

    axs = split_axes(axs)

    # Modify axis settings
    for ax, yr in zip(axs,years):
        ax.set_title(yr, y=1.0, pad=-14)
        # Remove ticks outside of data range
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(df.index.month.unique().min(),df.index.month.unique().max()+1,1)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.tick_params(axis='x', labelrotation=45)
        #ax.set_ylim((0.,100.))

    axs[0].legend(loc='upper center', bbox_to_anchor=(0.51,0.94)) # (offset-from-left, offset-from-bottom)
    axs[1].set_xlabel('Time (1st of month)', fontsize='x-large') # Time in YYYY-MM
    if plot_key == 'valid_observations_per_day':
        axs[0].set_ylabel('Valid meltwater observations \nper day (in % of total image)', fontsize='x-large')
    elif plot_key == 'total_meltwater_per_day':
        axs[0].set_ylabel('Total observed surface melt-\nwater over study area in '+r'km$^2$', fontsize='x-large')
    elif plot_key == 'average_meltwater_per_day':
        axs[0].set_ylabel('Average meltwater fraction \nper observed pixel', fontsize='x-large')

    filepath_to_save = f"{dir_figures}avg_meltwater_preds_per_day_{split}.png"
    if filepath_to_save is not None:
        Path(filepath_to_save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath_to_save)

    plt.show()
    plt.close()

    """
    # Plot number of valid pixels over time 
    fig, axs = plt.subplots(2, 1, figsize=(10,6), dpi=200)
    [axs[1].plot(df.index[t], df['valid_observations_per_day'].iloc[t], marker='X', markerfacecolor='None', color='black', alpha=alphas[t]) for t in range(n_days)]
    axs[1].plot(df.index, df['valid_observations_per_day'], color='black', alpha=0.7) 
    axs[1].tick_params(axis='x', labelrotation=45)
    axs[1].set_ylim((0.,1.))
    axs[1].set_xlabel('Time in years') # Time in YYYY-MM
    axs[1].set_ylabel('ratio valid / \n total pixels')
    """
    return 1

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate the quality of all predictions from different models.')
    parser.add_argument('--parallel', action='store_true', default=False, help='Enable parallel training')
    parser.add_argument('--verbose', type=bool, default=False, help='Set true to print verbose logs')
    parser.add_argument('--exclude_ssim', action='store_true', help='For debugging w/o ssim as it takes excessive memory')
    parser.add_argument('--data_split', type=str, default='val', help='Split [train, val, or test] for which the'\
                        'benchmark metrics will be calculcated')
    parser.add_argument('--compute_metrics', action='store_true', default=False, help='If true, computes the metrics on all predictions')
    parser.add_argument('--plot_predictions_vs_targets', action='store_true', default=False, help='If true, plots all model predictions vs the target')
    parser.add_argument('--plot_errors_vs_targets', action='store_true', default=False, help='If true, plots the error (preditions - targets) for each model vs the target')
    parser.add_argument('--plot_meltwater_extent_over_time', action='store_true', default=False, help='If true, plots the integrated meltwater content over time for each model and target')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for parallel processing')
    parser.add_argument('--unet', action='store_true', default=False, help='Add unet to evaluation')
    parser.add_argument('--time_interpolate_sar', action='store_true', default=False, help='Add time_interpolate_sar to evaluation')
    parser.add_argument('--interpolate_mar', action='store_true', default=False, help='Add interpolate_mar to evaluation')
    parser.add_argument('--linear_dem', action='store_true', default=False, help='Add linear_dem to evaluation')
    parser.add_argument('--deeplabv3', action='store_true', default=False, help='Add deeplabv3 to evaluation')
    parser.add_argument('--unet_smp', action='store_true', default=False, help='Add unet_smp to evaluation')
    parser.add_argument('--threshold_pmw', action='store_true', default=False, help='Add threshold_pmw to evaluation')
    parser.add_argument('--random_forest', action='store_true', default=False, help='Add random_forest to evaluation')
    parser.add_argument('--unet_smp_all', action='store_true', default=False, help='Add unet_smp_all to evaluation')
    parser.add_argument('--unet_smp_no_pmw', action='store_true', default=False, help='Add unet_smp_no_pmw to evaluation')
    parser.add_argument('--unet_smp_no_mar', action='store_true', default=False, help='Add unet_smp_no_mar to evaluation')
    parser.add_argument('--unet_smp_no_sar', action='store_true', default=False, help='Add unet_smp_no_sar to evaluation')
    parser.add_argument('--unet_smp_no_dem', action='store_true', default=False, help='Add unet_smp_no_dem to evaluation')
    parser.add_argument('--unet_smp_only_pmw', action='store_true', default=False, help='Add unet_smp_only_pmw to evaluation')
    parser.add_argument('--unet_smp_only_mar', action='store_true', default=False, help='Add unet_smp_only_mar to evaluation')
    parser.add_argument('--unet_smp_only_sar', action='store_true', default=False, help='Add unet_smp_only_sar to evaluation')
    parser.add_argument('--unet_smp_only_dem', action='store_true', default=False, help='Add unet_smp_only_dem to evaluation')
    parser.add_argument('--path_benchmark_figures', type=str, default='./references/figures/benchmark/', 
        help='Directory to store all plots from benchmarking the models')
    parser.add_argument('--data_root', type=str, 
        default='/home/gridsan/lutjens/EarthIntelligence_shared/datasets/hrmelt/raw/Helheim_data/reprojected_100m', 
        help='Directory that acts as data_root')
    return parser.parse_args()

if __name__ == '__main__':
    # Get command line arguments
    args = get_args()
    gdal.UseExceptions() # Enable gdal error messages (otherwise a warning is raised)
    
    # Initialize config, mainly with dataset paths
    cfg = {}
    cfg['verbose'] = args.verbose
    cfg['data_root'] = args.data_root
    cfg['path_val_split_csv'] = f'./runs/unet/data_v1_4/config/val.csv' # Image filenames used during validation
    cfg['path_test_split_csv'] = f'./runs/unet/data_v1_4/config/test.csv' # Image filenames used during test
    cfg['path_all_split_csv'] = f'./runs/unet/data_v1_4/config/all.csv' # Image filenames used during test
    cfg['split_cfg'] = 'csv'
    cfg['create_csv_w_split_cfg'] = False
    cfg['path_melt'] = 'SAR/S1Xv1.2/HH_PercentMelt/'
    cfg['path_landmask'] = 'GIMP_Mask/landMask_100m.tif'
    cfg['dtype'] = 'float32' # datatype for in-, output and model weights
    cfg['normalize_melt'] = False # don't normalize meltwater targets or predictions.
    cfg['path_benchmark_figures'] =  args.path_benchmark_figures

    # Set directories of every model's predictions
    # model_keys = ['interpolate_mar', 'time_interpolate_sar', 'deeplabv3', 'unet']
    # For time: model_keys = ['interpolate_mar', 'unet', 'time_interpolate_sar', 'linear_dem', 'deeplabv3']
    model_keys = []
    model_labels = {}
    if args.interpolate_mar:
        model_keys.append('interpolate_mar')
        model_labels['interpolate_mar'] = 'Interpolate MAR'
        cfg['path_predictions_interpolate_mar'] = Path('./runs/interpolate_mar/data_v1_4/predictions/')
    if args.linear_dem:
        model_keys.append('linear_dem')
        model_labels['linear_dem'] = 'Threshold DEM'
        cfg['path_predictions_linear_dem'] = Path('./runs/linear_dem/data_v1_4/predictions/')
    if args.threshold_pmw:
        model_keys.append('threshold_pmw')
        model_labels['threshold_pmw'] = 'Threshold PMW'
        cfg['path_predictions_threshold_pmw'] = Path('./runs/threshold_pmw/data_v1_4/predictions/')
    if args.time_interpolate_sar:
        model_keys.append('time_interpolate_sar')
        model_labels['time_interpolate_sar'] = 'Time-interpolate SAR'
        # cfg['path_predictions_time_interpolate_sar'] = Path('./runs/time_interpolate_sar/data_v1_4/predictions/')
        cfg['path_predictions_time_interpolate_sar'] = Path('/home/gridsan/lutjens/EarthIntelligence_shared/datasets/hrmelt/interim/runs/time_interpolate_sar/data_v1_4/predictions/')
    if args.random_forest:
        model_keys.append('random_forest')
        model_labels['random_forest'] = 'Random Forest'
        cfg['path_predictions_random_forest'] = Path('./runs/random_forest/data_v1_4/predictions/')
    if args.unet_smp:
        model_keys.append('unet_smp')
        model_labels['unet_smp'] = 'UNet'
        cfg['path_predictions_unet_smp'] = Path('./runs/unet_smp/data_v1_4/predictions/')
    if args.deeplabv3:
        model_keys.append('deeplabv3')
        model_labels['deeplabv3'] = 'DeepLabv3+'
        cfg['path_predictions_deeplabv3'] = Path('./runs/deeplabv3/data_v1_4/predictions/')
    if args.unet:
        model_keys.append('unet')
        model_labels['unet'] = 'vanilla UNet'
        cfg['path_predictions_unet'] = Path('./runs/unet/data_v1_4/predictions/')
    if args.unet_smp_all:
        model_keys.append('unet_smp_all')
        model_labels['unet_smp_all'] = 'UNet all channels'
        cfg['path_predictions_unet_smp_all'] = Path('./runs/unet_smp/data_v1_4_sensitivity/predictions/')
    if args.unet_smp_no_pmw:
        model_keys.append('unet_smp_no_pmw')
        model_labels['unet_smp_no_pmw'] = 'no PMW'
        cfg['path_predictions_unet_smp_no_pmw'] = Path('./runs/unet_smp/data_v1_4_sensitivity/no_pmw/predictions/')
    if args.unet_smp_no_mar:
        model_keys.append('unet_smp_no_mar')
        model_labels['unet_smp_no_mar'] = 'no MAR'
        cfg['path_predictions_unet_smp_no_mar'] = Path('./runs/unet_smp/data_v1_4_sensitivity/no_mar/predictions/')
    if args.unet_smp_no_sar:
        model_keys.append('unet_smp_no_sar')
        model_labels['unet_smp_no_sar'] = 'no SAR'
        cfg['path_predictions_unet_smp_no_sar'] = Path('./runs/unet_smp/data_v1_4_sensitivity/no_sar/predictions/')
    if args.unet_smp_no_dem:
        model_keys.append('unet_smp_no_dem')
        model_labels['unet_smp_no_dem'] = 'no DEM'
        cfg['path_predictions_unet_smp_no_dem'] = Path('./runs/unet_smp/data_v1_4_sensitivity/no_dem/predictions/')
    if args.unet_smp_only_pmw:
        model_keys.append('unet_smp_only_pmw')
        model_labels['unet_smp_only_pmw'] = 'only PMW'
        cfg['path_predictions_unet_smp_only_pmw'] = Path('./runs/unet_smp/data_v1_4_sensitivity/only_pmw/predictions/')
    if args.unet_smp_only_mar:
        model_keys.append('unet_smp_only_mar')
        model_labels['unet_smp_only_mar'] = 'only MAR'
        cfg['path_predictions_unet_smp_only_mar'] = Path('./runs/unet_smp/data_v1_4_sensitivity/only_mar/predictions/')
    if args.unet_smp_only_sar:
        model_keys.append('unet_smp_only_sar')
        model_labels['unet_smp_only_sar'] = 'only SAR'
        cfg['path_predictions_unet_smp_only_sar'] = Path('./runs/unet_smp/data_v1_4_sensitivity/only_sar/predictions/')
    if args.unet_smp_only_dem:
        model_keys.append('unet_smp_only_dem')
        model_labels['unet_smp_only_dem'] = 'only DEM'
        cfg['path_predictions_unet_smp_only_dem'] = Path('./runs/unet_smp/data_v1_4_sensitivity/only_dem/predictions/')
    if not model_keys:
        print('Warning: need to supply at least one model key as command line argument, e.g., --unet. For now, we evaluate only unet.')
        model_keys.append('unet')

    cfg['paths_predictions'] = []
    for model_key in model_keys:
        print(f'Retrieving {model_key} predictions from {cfg[f"path_predictions_{model_key}"]}')
        cfg['paths_predictions'].append(cfg[f'path_predictions_{model_key}'])

    # Init compute parameters, e.g., cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg['num_workers'] = set_num_workers(num_workers=None, parallel=True)
    cfg['batch_size'] = args.batch_size
    cfg['prediction_batch_size'] = args.batch_size
    cfg['seed'] = 42
    print(f'Using device {device}, num workers: {cfg["num_workers"]}, batch size: {cfg["batch_size"]}')
    
    # Dictionary with all desired metrics and the function to compute them
    metrics_fn = {
        'MaskedL1': MaskedLoss('l1', reduction='none'),
        'MaskedMSE': MaskedLoss('mse', reduction='none'),
        'MaskedR2': MaskedR2(reduction='none'),
        'ValidPx': CountValidPx(reduction='none'),
        'MaskedSSIM': MaskedSSIM(data_range = (0.,1.), sigma=10., device=device),
        'MaskedAccuracy': MaskedLoss('accuracy', reduction='none', threshold=0.1),
        'MaskedPrecision': MaskedPrecision(reduction='none', threshold=0.1),
        'MaskedRecall': MaskedRecall(reduction='none', threshold=0.1),
    }
    if args.exclude_ssim and 'MaskedSSIM' in metrics_fn:
        del metrics_fn['MaskedSSIM']

    # Compute all metrics
    if args.compute_metrics:
        metrics = benchmark_metrics(metrics_fn, model_keys, device, cfg, split=args.data_split, verbose=False)

    if args.plot_predictions_vs_targets:
        plot_predictions_vs_targets(model_keys, device, cfg, dpi=600, split=args.data_split)

    if args.plot_errors_vs_targets:
        plot_predictions_vs_targets(model_keys, device, cfg, dpi=300, plot_errors=True, split=args.data_split)

    if args.plot_meltwater_extent_over_time:
        plot_meltwater_extent_over_time(model_keys, device, cfg, split=args.data_split, model_labels=model_labels)

    #  For each model, create a plot of unnormalized model input, model prediction, model target, error pred-target
