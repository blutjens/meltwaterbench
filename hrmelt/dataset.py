import os
import re
import argparse
import yaml
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import torch
from torch.utils.data import Dataset, default_collate
from torch.utils.data import random_split
import torchvision.transforms as T # Compose, Resize, ToTensor, Normalize
from torchvision.transforms.functional import gaussian_blur 
from PIL import Image
from osgeo import gdal
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
from hrmelt.utils.utils import lookup_torch_dtype

def remove_invalid_pmw_entries(data, verbose, run_parallel=False):
    """
    Removes invalid pmw entries from data
    Returns
        data
    """
    from hrmelt.utils import parallel
    counter = 0
    len_data = len(data)

    def has_nans(data, counter, verbose):
        path = data[counter]['pmw']
        ds = gdal.Open(str(path))
        array = np.array(ds.GetRasterBand(1).ReadAsArray())
        del ds
        if np.isnan(np.sum(array)):
            return True
        else:
            return False

    logs = len_data*[None]
    has_nans_fn, tasks = parallel.init_preprocessing(fn=has_nans, 
                                                     parallel=run_parallel,
                                                     tmpdir='/tmp/ray/',
                                                     slurm=True)
    for idx in tqdm(range(len_data)):
        fn_args = {'data':data,'counter':idx,'verbose':verbose}

        if not run_parallel:
            logs[idx] = has_nans_fn(**fn_args)
        else:
            tasks.append(has_nans_fn(**fn_args))

    # Parse parallel tasks
    if run_parallel:
        tasks = parallel.get_parallel_fn(tasks)
        for i in range(len_data):
            logs[i] = tasks[i]

    # Remove all entries from data that contain nans
    data_valid = [elem for elem, is_invalid in zip(data, logs) if not is_invalid]
    data_invalid = [elem for elem, is_invalid in zip(data, logs) if is_invalid]
    if verbose:
        for elem in data_invalid:
            path = elem['pmw']
            ds = gdal.Open(str(path))
            array = np.array(ds.GetRasterBand(1).ReadAsArray())
            del ds
            invalid_area_in_pct = np.count_nonzero(np.isnan(array)) / np.prod(array.shape) * 100
            print(f'Removing entry, because PMW on {path.stem} has {invalid_area_in_pct:f}% invalid area.')

    return data_valid

def remove_invalid_entries(data: List[Dict], verbose:bool = False,
                           run_parallel:bool = False) -> List[Dict]:
    """
    Checks the list of filepaths and removes invalid entries according to some
    criteria

    Args:
        run_parallel: If True, uses ray to parallelize the code across cpu
    """
    if 'pmw' in data[0].keys():
        if verbose: 
            print('Opening every pmw image to check if it contains any NaNs. If so, the data '\
                  'is removed from the stack. To avoid set cfg.check_for_invalid_entries=False')
        data = remove_invalid_pmw_entries(data, verbose, run_parallel=run_parallel)
    return data

def get_filepaths_for_deployment(cfg: Dict, verbose:bool = False) -> List[Dict]:
    """
    Creates a list of filepaths from every possible day where all input data
    streams are available. 'melt' targets are not added to the data stack. 
    We assume that files are uniquely indexed by year_month_date.tif. Every

        Returns:
        complete_data_entries: 
            List[Dict(  'in_key1': path,
                        'in_key2': path,
                        ... 
                        'in_keyn': path),...,Dict()]
    """
    data_root = cfg['data_root']
    in_keys = cfg['in_keys']

    # Define max range over which to search for input keys
    deploy_start_time = cfg['deploy_start_time'] if 'deploy_start_time' in cfg else '2017_04_01'
    deploy_end_time = cfg['deploy_end_time'] if 'deploy_end_time' in cfg else '2023_09_30'
    start_timestamp = datetime.datetime.strptime(deploy_start_time, "%Y_%m_%d")
    end_timestamp = datetime.datetime.strptime(deploy_end_time, "%Y_%m_%d")

    complete_data_entries = []

    def iterate_dates(start_t, end_t):
        """
        Returns all days between start and end date. E.g.,
        with start 2017_04_01 and end 2023_09_30 returns
        all dates from 2017 to 2023; April to Sept; and 1st
        to 30th.
        """
        current_date = start_t
        while current_date <= end_t:
            if (start_t.month <= current_date.month <= end_t.month and 
                start_t.day <= current_date.day <= end_t.day):
                yield current_date
            current_date += datetime.timedelta(days=1)

    # Iterate over every day in range and add day to complete_
    #  _data_entries if all in_keys are available on that day.
    for date in iterate_dates(start_timestamp, end_timestamp):
        data_stack = {}
        for in_key in in_keys:
            filepath_in_key = Path(data_root) / Path(cfg['path_' + in_key]) / Path(date.strftime("%Y_%m_%d") + '.tif')
            if filepath_in_key.is_file():
                data_stack[in_key] = filepath_in_key
        # Add complete data entries to dataset list
        if len(data_stack) == len(in_keys): 
            complete_data_entries.append(data_stack)
        elif verbose:
            print(f'Incomplete entry. Did not find: {in_keys - data_stack.keys()}. Skipping {date.strftime("%Y_%m_%d")}')

    return complete_data_entries

def get_filepaths_of_complete_entries(cfg: Dict, verbose:bool = False) -> List[Dict]:
    """
    Creates a list of filepaths of complete data entries. A data entry is 
    complete if for a given output image every required input variable exists. 
    Here, we check alignment by assuming that all filenames are year_month_date , 
    e.g. 2018_06_03.tif. Every entry has a unique ID which in this case is the 
    time stamp.

    Returns:
        complete_data_entries: 
            List[Dict(  'in_key1': path,
                        'in_key2': path,
                        ... 
                        'melt': path),...,Dict()]
    """
    data_root = cfg['data_root']
    in_keys = cfg['in_keys']

    complete_data_entries = []

    # Retrieve all filenames
    im_paths = {}
    for in_key in in_keys:
        dir_in_key = str(data_root/Path(cfg['path_' + in_key]))
        im_paths[in_key] = get_filepaths_in_dir(dir_in_key, cfg['filetype'])
    dir_melt = str(data_root/Path(cfg['path_melt']))
    print('Looking for data in: ', dir_melt)
    im_paths['melt'] = get_filepaths_in_dir(dir_melt, cfg['filetype']) 

    # Collect all complete entries
    # Adds path of each input variable, iff the timestamps match with the melt data
    for idx, im_path_melt in enumerate(im_paths['melt']):
        data_stack = {}

        # Check if output image (meltwater) follows naming convention
        time_melt = im_path_melt.stem
        assert bool(re.match(r'\d{4}_\d{2}_\d{2}', time_melt)), ''\
            'Filename convention for surface meltwater tiles is '\
            'year_month_day.tif, e.g., 2018_06_03.tif' 

        for in_key in in_keys: 
            # Retrieve the image path with same timestamp
            first_matching_path = None
            for im_path_in in im_paths[in_key]:
                if time_melt in str(im_path_in) and first_matching_path is None:
                    first_matching_path = im_path_in
                    data_stack[in_key] = first_matching_path
        
        # Add complete data entries to dataset list
        if len(data_stack) == len(in_keys): 
            data_stack['melt'] = im_path_melt
            complete_data_entries.append(data_stack)
        elif verbose: 
            print(f'Incomplete entry. Did not find: {in_keys - data_stack.keys()}. Skipping {im_path_melt.stem}')

    return complete_data_entries

def get_filepaths_in_dir(directory: str, filetype='.tif') -> List:
    paths = Path(directory).glob('**/*'+filetype)
    filepaths = [x for x in paths if x.is_file()]
    filepaths.sort()
    return filepaths

def get_random_crop(path: str, img_size: [int, int]) -> [int,int]:
    """
    Calculates random offsets to open image. If img_size < actual image 
    size, the full image is returned.
    Source: edited from ChatGPT

    Inputs:
        path
        img_size  [height, width]
    Returns:
        offsets [height, width]
        img_size  [height, width]
    """
    ds = gdal.Open(path)

    # Get the size of the image
    width = ds.RasterXSize
    height = ds.RasterYSize
    del ds

    # Specify the size of the random crop
    crop_height = min(img_size[0], height) 
    crop_width = min(img_size[1], width)

    # Calculate the maximum offset to prevent going out of bounds
    max_y_offset = height - crop_height
    max_x_offset = width - crop_width

    # Generate random offsets for the crop. 
    # Note: using torch.randint instead of np.random for torch
    #  to handle the random seeds.
    y_offset = torch.randint(0, max_y_offset+1, (1,)).cpu().numpy().astype(int)[0].item()
    x_offset = torch.randint(0, max_x_offset+1, (1,)).cpu().numpy().astype(int)[0].item()
    return [y_offset, x_offset], [crop_height, crop_width]

def open_cropped_tif(path: str, 
    img_size: [int,int] = None, 
    offsets: [int, int] = None) -> np.ndarray:
    """
    Returns a cropped image from a given .tif. Only loads the
    cropped image into memory; not the full .tif. 

    Input:
        img_size [height, width]
        offsets [y_offset, x_offset]
    Returns:
        array (height, width, n_channels)
    """
    ds = gdal.Open(path)
    if img_size is None:
        array = np.array(ds.GetRasterBand(1).ReadAsArray())
    else:
        if offsets is None:
            offsets, img_size = get_random_crop(path)

        # Read the cropped area using GDAL
        array = ds.GetRasterBand(1).ReadAsArray(
            offsets[1], offsets[0], img_size[1], img_size[0])

    del ds
    return array[...,np.newaxis]

def get_modis_b_mean_std(modis_b, max_modis=10000):
    """
    Calculates mean and standard deviation after masking negative values
    and max scaling to [0,1].

    Input:
        modis_b: np.array(h,w) : A single band image
    Returns 
        mean np.array(1)
        std np.array(1)
    """
    mask = modis_b <= -1 # mask all negative values
    modis_b = modis_b.astype(np.float32) # note: src is int16, so converting to float32 has overhead
    modis_b = modis_b / float(max_modis)
    mean = np.ma.array(modis_b, mask=mask).mean()
    std = np.ma.array(modis_b, mask=mask).std()
    return mean, std

def get_filepaths_from_csv(path_csv, 
    data_root='/mnt/c/Users/Bjoern/code/hrmelt/data/raw/reprojected_100m/', 
    in_keys=None,
    subpaths={
        'path_pmw': 'PMW/',
        'path_mar_wa1': 'MAR/MARv3.12/WA1/',
        'path_melt': 'SAR/S1A/HH_PercentMelt/'
    },
    sort=False,
    data_keys=['pmw', 'mar_wa1', 'melt'],):
    """
    Reads in a csv file of image filenames and converts it to a data 
    stack of filepaths. The image filename is assumed to be a unique
    identifier across data variables. The csv file is assumed to be
    a list of filenames with the format YYYY_MM_DD.tif, e.g.: 
    2018_05_01.tif
    2018_05_04.tif
    ...
    2020_02_09.tif
        
    Args:
        path_csv str: Path to csv file of train, val, or test split
        data_root str: Path to data rootfolder
        in_keys List['in_key1', 'in_key2', ...]: Keys to in- and out-
            put channels. Note, due to backcompability in_keys contains
            'melt' in this function. 
        subpaths Dict('in_key1': subpath1,
                'in_key2': subpath2,
                ...
                'melt': subpath_melt): Paths tyo variables that are used
                    as in- or outputs.
        sort bool: If True, the returned data will be sorted by timestamp
    Returns:
        data: 
            List[n_samples * Dict(  'in_key1': path,
                        'in_key2': path,
                        ... 
                        'melt': path),...,Dict()]
    """
    if in_keys is not None:
        raise AttributeError('In function dataset.py -> get_filepaths_from_csv; '\
                            'the argument in_keys is deprecated. Use data_keys '\
                            'instead which is in_keys + "melt" if the dataloader'\
                            'should return targets')
    # Read the csv file using pandas
    csv_file = Path(path_csv)
    df = pd.read_csv(csv_file, header=None)
    filenames = df.squeeze('columns')

    # Sort the filenames by timestamp
    if sort:
        filenames = filenames.sort_values(ignore_index=True)

    # Add the filepaths of every variable as 
    data = []
    for filename in filenames:
        filepaths = dict()
        for key in data_keys:
            # Get full filepath from filename
            if 'predictions' in key:
                # Assuming that paths with key predictions_unet, predictions_deeplabv3, etc. 
                #  are predictions from trained models and full path is given. 
                dir_key = Path(subpaths[f'path_{key}'])
            else:
                dir_key = Path(data_root)/Path(subpaths[f'path_{key}'])
            filepaths[key] = dir_key/Path(filename)
        data.append(filepaths)

    return data, filenames

def batch_preprocess_fn(batch, cfg, device):
    """
    This function applies gausian blur to the batch which leads to better performance.
    The function is injected into the dataloader via the collate_fn param.
    To use this in the dataloader cfg and device has to be defined via partial.
    Args:
        batch (list): structure: ->
            list(of size batch)[tuple(input -> (c,h,w), target_tensor -> (c,h,w), mask_tensor -> (c,h,w), metadata -> dict)]
        cfg dict: Run config
        device str: Device to be used in the pre process.
    Returns:
        batch arr: Processed batch tensor structure: ->
            list[input -> (b,c,h,w), target -> (b,c,h,w), mask -> (b,c,h,w), metadata -> dict]
    """    
    # used to keep default collate structure see -> return param
    # Before collate, batch is a list of length batch_size with each element being a tuple that contains __getitem__(), e.g., 4, Tensors [(Tensor, Tensor, Tensor, Tensor), (...)]; 
    # After collate, batch is a list of length __getitem()__ with each element being a Tensor of shape [B, C, H, W]
    # Before and after, both are on the 'cpu'
    batch = default_collate(batch)

    device_prev = batch[0].device # likely, 'cpu'

    if cfg['use_batch_blur']:
        inputs = batch[0].to(device) # send to GPU for batch processing.

        # Find the indices of 'pmw' and 'mar_wa1' in cfg['in_keys'], if they exist
        pmw_idx = cfg['in_keys'].index('pmw') if 'pmw' in cfg['in_keys'] else None
        mar_idx = cfg['in_keys'].index('mar_wa1') if 'mar_wa1' in cfg['in_keys'] else None

        # Process 'pmw' if it exists
        if pmw_idx is not None:
            # select input
            inputs[:,pmw_idx] = gaussian_blur(img=inputs[:,pmw_idx],
                kernel_size=int(cfg['pmw_GaussianBlur_kernel_size']),
                sigma=cfg['pmw_GaussianBlur_sigma'])
            # inputs[:,pmw_idx] = T.GaussianBlur(
            #     kernel_size=cfg['pmw_GaussianBlur_kernel_size'],
            #     sigma=cfg['pmw_GaussianBlur_sigma'])(inputs[:,pmw_idx])
            
            #pmw_batch = batch[0][:, pmw_idx].to(device)
            #pmw_batch = T.GaussianBlur(
            #    kernel_size=cfg['pmw_GaussianBlur_kernel_size'],
            #    sigma=cfg['pmw_GaussianBlur_sigma'])(pmw_batch)
            #pmw_batch = pmw_batch.to(device_prev)
            #batch[0][:, pmw_idx] = pmw_batch
            #del pmw_batch

        # Process 'mar_wa1' if it exists
        if mar_idx is not None:
            inputs[:,mar_idx] = gaussian_blur(img=inputs[:,mar_idx],
                kernel_size=int(cfg['pmw_GaussianBlur_kernel_size']),
                sigma=cfg['mar_wa1_GaussianBlur_sigma'])
            
            #mar_batch = batch[0][:, mar_idx].to(device)
            #mar_batch = T.GaussianBlur(
            #    kernel_size=cfg['mar_wa1_GaussianBlur_kernel_size'],
            #    sigma=cfg['mar_wa1_GaussianBlur_sigma'])(mar_batch)
            #mar_batch = mar_batch.to(device_prev)
            #batch[0][:, mar_idx] = mar_batch
            #del mar_batch
        
        batch[0] = inputs.to(device=device_prev)
        # torch.cuda.empty_cache()
        del inputs

    return batch

class HRMeltDataset(Dataset):
    def __init__(self, cfg, split='train', verbose=False, check_data_in_parallel=False):
        '''
            Constructor. Here, we collect and index the dataset inputs and
            labels. 
            Data should follow the below format. The filename IDs are used to 
            match input modilities across folders.
            - Path: cfg.data_root/cfg.path_<in_key>
            - Filename: unique_id.tif, here, <year>_<month>_<date>, e.g., 2018_06_03.tif
        Args:
            check_data_in_parallel bool: If True, selected data checks are parallelized
                across cpu using ray.
        '''
        self.cfg = cfg
        self.split = split
        self.verbose = verbose

        # Index data into list
        self.data = []
        if self.cfg['split_cfg'] == 'csv':
            # Retrieve datapoints from a .csv file, assuming that those datapoints 
            # have already been checked for incomplete and invalid entries

            # Concatenate keys of all in- and output channels
            data_keys = self.cfg['in_keys'] + ['melt']
            if self.split == 'deploy':
                data_keys.remove('melt')

            subpaths = {} # subselect the cfg variables that contain a path_
            for in_key in data_keys:
                subpaths[f'path_{in_key}'] = self.cfg[f'path_{in_key}']

            self.data, self.filenames = get_filepaths_from_csv(
                path_csv=self.cfg[f'path_{split}_split_csv'],
                data_root=self.cfg['data_root'],
                data_keys=data_keys,
                subpaths=subpaths)
        else:
            if self.split != 'deploy':
                # Retrieve datapoints that contain all desired in- and output channels; this might take a while
                self.data = get_filepaths_of_complete_entries(self.cfg, verbose=self.verbose)
            elif self.split == 'deploy':
                # Get paths for deploying the trained model without any targets
                self.data = get_filepaths_for_deployment(self.cfg, verbose=self.verbose)

            cfg['check_for_invalid_entries'] = cfg['check_for_invalid_entries'] if 'check_for_invalid_entries' in cfg else True
            if cfg['check_for_invalid_entries']:
                self.data = remove_invalid_entries(self.data, verbose=self.verbose, run_parallel=check_data_in_parallel)

        # Dtype of every returned item.
        self.dtype = lookup_torch_dtype(self.cfg['dtype'])
        assert self.dtype == torch.float32, f'Dataset currently only accept float32, but got {self.dtype}'
        # Keys of __getitem__ return
        if 'in_keys_aux' not in self.cfg.keys():
            self.cfg['in_keys_aux'] = []
        self.keys = self.cfg['in_keys'] + self.cfg['in_keys_static'] + self.cfg['in_keys_aux'] + ['melt_mask', 'melt']

        # Set default normalization config
        if 'normalize_inputs' not in self.cfg.keys():
            self.cfg['normalize_inputs'] = True
        if 'normalize_melt' not in self.cfg.keys():
            self.cfg['normalize_melt'] = False
        if 'use_batch_blur' not in self.cfg.keys():
            self.cfg['use_batch_blur'] = False

        # If split modality is time, we split self.data into train, val, test
        # Initialize data transformation and augmentation
        # self.transform = T.Compose([])

    def __len__(self):
        '''
            Returns the length of the dataset.
        '''
        return len(self.data)

    def sort(self):
        """
        Sorts the dataset by meltwater image timestamp
        """                
        # Create a list of tuples combining data entries and filenames
        pairs = list(zip(self.data, self.filenames))

        # Sort the entries based on the filenames
        pairs.sort(key=lambda x: Path(x[1]).stem)

        # Separate the sorted elements and indices
        self.data, self.filenames = zip(*pairs)

        # sample_key = list(self.data[0].keys())[0]
        # Old:
        # self.data.sort(key=lambda x: x['melt'].stem)

    def create_data_splits(self, 
            split_cfg: str = 'day',
            val_percent: float = 0.1,
            test_percent: float = 0.0,
            seed: int = 0,
            create_csv_w_split_cfg: bool = False,
            path_train_split_csv: str = None,
            path_val_split_csv: str = None,
            path_test_split_csv: str = None,
            path_all_split_csv: str=None,
            n_imgs_per_month: int = 2, 
            ):
        '''
            Creates data splits across train, val, test.
            Args:
                val_percent: Percentage of validation data, if split_cfg=='day'
                split_cfg: Desired split configuration, if split_cfg=='day'
                seed: Seed for random splits
                create_csv_w_split_cfg: If True, this will save and overwrites the created split cfg as a csv
                path_train_split_csv: If create_csv_w_split_cfg, the split config will stored in this path as csv with image filenames
                path_val_split_csv: Same as path_train_split_csv, but for the validation split
                path_test_split_csv: Same as path_train_split_csv, but for the test split
                path_all_split_csv: If not None, will create a .csv with all filenames
                n_imgs_per_month: If split_cfg=='stratified_time', this sets the number of images in the test and val set of each month of each year
        '''
        # Sort by timestamp
        self.sort()

        if split_cfg == 'space':
            raise NotImplementedError('split across space is not implemented.')
        elif split_cfg == 'day':
            # Random split will split randomly across time using the year_month_day timestamp as UID
            n_test = int(len(self) * test_percent) # int() always rounds down.
            n_val = int(len(self) * val_percent)
            n_train = len(self) - n_val - n_test

            train_set, val_set, test_set = random_split(
                self, [n_train, n_val, n_test],
                generator=torch.Generator().manual_seed(seed))

            if create_csv_w_split_cfg:
                # Store the split configuration of every split as csv that contains the filenames
                all_filenames = []
                for subset, path_split_csv in zip(
                    (train_set, val_set, test_set), 
                    (path_train_split_csv, path_val_split_csv, path_test_split_csv)):

                    # Extract list of filenames from dataset
                    idx_meta = -1 # Index to meta information in dataset.__getitem__
                    filenames = []
                    for item in subset:
                        filenames.append(Path(item[idx_meta]['path_melt']).name) # e.g., '2018_04_15.tif'
                    all_filenames.extend(filenames)

                    # Store list of filenames as a csv using pandas
                    series = pd.Series(filenames)
                    Path(path_split_csv).parent.mkdir(parents=True, exist_ok=True) # Create parent directory, if not exist
                    print('Saving .csv at: ', path_split_csv)
                    series.to_csv(path_split_csv, header=False, index=False)
                
                # save additional.csv that contains all filenames
                if path_all_split_csv is not None:
                    series = pd.Series(all_filenames)
                    Path(path_all_split_csv).parent.mkdir(parents=True, exist_ok=True) # Create parent directory, if not exist
                    series.to_csv(path_all_split_csv, header=False, index=False)
        elif split_cfg == 'stratified_time':
            # Create a stratified data split across time that contains n_imgs_per_month images per month.

            # Get timestamps
            paths_melt = [data_entry['melt'] for data_entry in self.data] # e.g., '/path/to/.../2017_08_23.tif
            timestamps = [Path(path_melt).stem for path_melt in paths_melt] # e.g., ['2017_08_23',...]
            timestamps = [datetime.datetime.strptime(timestamp, "%Y_%m_%d") for timestamp in timestamps]

            # Convert self.data from list into dataframe to be able to filter by timestamp
            df = pd.DataFrame(index=timestamps)
            for key in self.data[0].keys():
                df[key] = [data_entry[key] for data_entry in self.data]

            def pick_n_images_from_df(df, n_imgs_per_month):
                # Assemble test and val set by iterating over every month in every year and 
                #  picking n_imgs_per_month from it.
                df_newset = pd.DataFrame()
                for year in df.index.year.unique():
                    df_yr = df[df.index.year == year]
                    if year == 2017:
                        # Skipping 2017 dataset, because we there's not enough observations
                        continue
                    else:
                        for month in df_yr.index.month.unique():
                            df_mon = df_yr[df_yr.index.month == month]
                            # Draw n random samples
                            df_sample = df_mon.sample(n=n_imgs_per_month)
                            # Remove samples from full dataset
                            df = df.drop(df_sample.index)
                            # Add samples to newly created set
                            df_newset = pd.concat([df_newset, df_sample])
                return df_newset, df

            df_all = df.copy()
            df_test, df = pick_n_images_from_df(df, n_imgs_per_month)
            df_val, df = pick_n_images_from_df(df, n_imgs_per_month)
            df_train = df

            if create_csv_w_split_cfg:
                # Store list of filenames as csv
                for df_split, path_split_csv in zip(
                    (df_train, df_val, df_test, df_all), 
                    (path_train_split_csv, path_val_split_csv, path_test_split_csv, path_all_split_csv)):
                    filenames = df_split.index.strftime('%Y_%m_%d.tif').to_series()
                    Path(path_split_csv).parent.mkdir(parents=True, exist_ok=True) # Create parent directory, if not exist
                    filenames.to_csv(path_split_csv, header=False, index=False)

            # Convert dataframe back to list of dictionaries
            test_set = df_test.to_dict('records')
            val_set = df_val.to_dict('records')
            train_set = df_train.to_dict('records')

        return train_set, val_set, test_set

    def __getitem__(self, idx, offsets=None):
        '''
            Returns a single datastack of tiles. The returned tiles come from a large tif 
            where only a crop of size, cfg.img_size, at a random location is loaded. All
            data is converted to float32.

            Args:
                idx int: index into list of tifs in self.data
                offsets [int, int]: offsets of top-left corner of tile of tif. Used by child
                 class HRMeltDatasetConvolution()
            Returns:
                inputs torch.Tensor(5, h, w): Inputs with PMW, MODIS band 1, 
                modis band 2, MODIS mask, MAR wa1
            targets torch.Tensor(1, h, w, ): Percentage of surface melt
            targets_mask torch.Tensor(1, h, w): Output mask 
                with 1. for invalid values and 0. for valid values.
            meta dict(): meta information on return datastack
        '''
        # Get datastack
        data_stack = self.data[idx]

        # Initialize inputs
        n_ch = len(self.cfg['in_keys']) + len(self.cfg['in_keys_static']) + len(self.cfg['in_keys_aux'])
        assert n_ch >= 1, 'Error. config["in_keys*"] need to have at least one key'
        height = self.cfg['img_size'][0]
        width = self.cfg['img_size'][1]
        inputs = torch.empty((n_ch, height, width), dtype=self.dtype)
        ch_idx = 0
        meta = {}
        if 'melt' in data_stack:
            meta['path_melt'] = str(data_stack['melt'])
        meta['channels'] = dict()
        meta['filename'] = self.filenames[idx]

        if not data_stack:
            # No in- or output channels, e.g., for time_interpolate_sar deployment
            inputs = torch.zeros((0, 0, 0), dtype=self.dtype).contiguous()
            melt = torch.zeros((0, 0, 0), dtype=self.dtype).contiguous()
            melt_mask = torch.zeros((0, 0, 0), dtype=self.dtype).contiguous() # All pixels valid.
            return inputs, melt, melt_mask, meta
        
        # Load and transform data stack
        if offsets is None:
            sample_key = list(data_stack.keys())[0]
            offsets, self.cfg['img_size'] = get_random_crop(str(data_stack[sample_key]), self.cfg['img_size'])

        # Adding PMW
        if 'pmw' in self.cfg['in_keys']:
            pmw = open_cropped_tif(str(data_stack['pmw']), self.cfg['img_size'], offsets)
            if self.verbose:
                print('Stats of cropped tile before scale and normalize.')
                print(f'pmw min, max, mean, std:   \t{pmw.min(), pmw.max(), pmw.mean(), pmw.std(), pmw.dtype.name}')
            assert pmw.dtype == np.float32, f'Expected dtype float32, but got {pmw.dtype} for pmw'
            pmw = torch.from_numpy(pmw.transpose((2, 0, 1))).contiguous() # Pytorch uses channels-first: (c, h, w)
            # We apply no scaling, because PMW does not have a defined lower/upper limit. (todo: double-check)
            if self.cfg['normalize_inputs']:
                pmw = T.Normalize(mean=self.cfg['mean_pmw'], std=self.cfg['std_pmw'])(pmw)
            
            if not self.cfg['use_batch_blur']:
                # Add GaussianBlur to smooth out the sharp edges in the incoming image.
                # If they're not smoothed out, I have seem them introduce edge artifacts.
                pmw = T.GaussianBlur(
                    kernel_size=self.cfg['pmw_GaussianBlur_kernel_size'],
                    sigma=self.cfg['pmw_GaussianBlur_sigma'])(pmw)

            inputs[ch_idx:ch_idx+1,...] = pmw
            meta['channels'][ch_idx] = 'pmw' 
            ch_idx += 1

        # Adding MODIS
        if 'modis_b1' in self.cfg['in_keys'] or 'modis_b2' in self.cfg['in_keys']:
            modis_b1 = open_cropped_tif(str(data_stack['modis_b1']), self.cfg['img_size'], offsets)
            modis_b2 = open_cropped_tif(str(data_stack['modis_b2']), self.cfg['img_size'], offsets)
            modis = np.concatenate((modis_b1, modis_b2), axis=-1)
            if self.verbose:
                modis_b1_mean, modis_b1_std = get_modis_b_mean_std(modis[...,0], max_modis=self.cfg['max_modis'])
                print(f'modis_b1 min, max, mean, std after masking: \t{modis[...,0].min(), modis[...,0].max(), modis_b1_mean, modis_b1_std, modis_b1.dtype.name}')
            mask_np = modis <= -1
            modis = np.ma.array(modis, mask=mask_np).filled(fill_value=0) # Fill masked values with zero.
            if self.verbose:
                modis_b1_mean, modis_b1_std = get_modis_b_mean_std(modis[...,0], max_modis=self.cfg['max_modis'])
                print(f'modis_b1 min, max, mean, std: \t{modis[...,0].min(), modis[...,0].max(), modis_b1_mean, modis_b1_std, modis_b1.dtype.name}')
            modis = modis.astype(np.float32) # note: src is int16, so converting to float32 has overhead
            modis = torch.from_numpy(modis.transpose((2, 0, 1))).contiguous() # convert to tensor
            modis = modis / float(self.cfg['max_modis']) # min-max scale
            # if self.cfg['normalize_inputs']:
            #     modis = T.Normalize(mean=[self.cfg['mean_modis_b1'], self.cfg['mean_modis_b2']],
            #         std=[self.cfg['std_modis_b1'], self.cfg['std_modis_b2']])(modis) # normalize to zero-mean, unit variance
            # todo: set all masked value to zero again

            # Adding MODIS mask
            # Todo: How to pass mask to CNN? -> For now, passing mask as extra channel and by setting all masked values in array to zero.
            modis_mask = np.ma.mask_or(mask_np[...,0], mask_np[...,1], shrink=False)[...,None] # Create union mask of both channels
            modis_mask = modis_mask.astype(np.float32) # adding overhead, but model only accepts float
            # from IPython.core.debugger import set_trace; set_trace()
            modis_mask = torch.from_numpy(modis_mask.transpose((2,0,1))).contiguous() # to torch

            inputs[ch_idx:ch_idx+2,...] = modis
            meta['channels'][ch_idx] = 'modis_b1' 
            meta['channels'][ch_idx+1] = 'modis_b2' 
            ch_idx += 2

        # Adding MAR. 
        if 'mar_wa1' in self.cfg['in_keys']:
            # mar_wa1 is expected to be in range [0,1].
            mar_wa1 = open_cropped_tif(str(data_stack['mar_wa1']), self.cfg['img_size'], offsets)
            if self.verbose:
                print(f'mar_wa1 min, max, mean, std:\t{mar_wa1.min(), mar_wa1.max(), mar_wa1.mean(), mar_wa1.std(), mar_wa1.dtype.name}')
            assert mar_wa1.dtype == np.float32, f'Expected dtype float32, but got {mar_wa1.dtype} for mar_wa1'
            mar_wa1 = torch.from_numpy(mar_wa1.transpose((2, 0, 1))).contiguous()
            mar_wa1 = mar_wa1 / float(self.cfg['max_mar_wa1'])
            # Though mar is expected in [0,1] range, the min, max, mean, std. is ~[0.01,0.04,0.07,0.01]
            #  so the values are very low and need to be normalized.
            if self.cfg['normalize_inputs']:
                mar_wa1 = T.Normalize(mean=self.cfg['mean_mar_wa1'], std=self.cfg['std_mar_wa1'])(mar_wa1)

            if not self.cfg['use_batch_blur']:
                # Add GaussianBlur to smooth out the sharp edges in the incoming image.
                mar_wa1 = T.GaussianBlur(
                    kernel_size=self.cfg['mar_wa1_GaussianBlur_kernel_size'],
                    sigma=self.cfg['mar_wa1_GaussianBlur_sigma'])(mar_wa1)

            inputs[ch_idx:ch_idx+1,...] = mar_wa1
            meta['channels'][ch_idx] = 'mar_wa1' 
            ch_idx += 1

        # Adding time_interpolate_sar predictions
        if 'time_interpolate_sar' in self.cfg['in_keys']:
            time_interpolate_sar = open_cropped_tif(str(data_stack['time_interpolate_sar']), self.cfg['img_size'], offsets)
            assert time_interpolate_sar.dtype == np.float32, f'Expected dtype float32, but got {time_interpolate_sar.dtype} for time_interpolate_sar'
            time_interpolate_sar = torch.from_numpy(time_interpolate_sar.transpose((2, 0, 1))).contiguous()
            inputs[ch_idx:ch_idx+1,...] = time_interpolate_sar
            meta['channels'][ch_idx] = 'time_interpolate_sar' 
            ch_idx += 1

        # Add static input features
        # Adding DEM
        if 'dem' in self.cfg['in_keys_static']:
            dem_filepath = Path(self.cfg['data_root'])/Path(self.cfg['path_dem'])
            dem = open_cropped_tif(str(dem_filepath), self.cfg['img_size'], offsets)
            if self.verbose:
                print(f'dem min, max, mean, std:   \t{dem.min(), dem.max(), dem.mean(), dem.std(), dem.dtype.name}')
            dem = dem.astype(np.float32)
            dem = torch.from_numpy(dem.transpose((2, 0, 1))).contiguous() # Pytorch uses channels-first: (c, h, w)
            if self.cfg['normalize_inputs']:
                dem = T.Normalize(mean=self.cfg['mean_dem'], std=self.cfg['std_dem'])(dem)

            inputs[ch_idx:ch_idx+1,...] = dem
            meta['channels'][ch_idx] = 'dem' 
            ch_idx += 1

        # Add targets
        # Add target surface meltwater
        if self.split != 'deploy':
            melt = open_cropped_tif(str(data_stack['melt']), self.cfg['img_size'], offsets)
            mask_nans = np.ma.masked_invalid(melt).mask # Mask nans from, e.g., overexposure
            melt = np.ma.array(melt, mask=mask_nans).filled(fill_value=0) # Fill masked values with zero.
            if self.verbose:
                melt_mean = np.ma.array(melt, mask=mask_nans).mean()
                melt_std = np.ma.array(melt, mask=mask_nans).std()
                print(f'melt min, max, mean, std:   \t{melt.min(), melt.max(), melt_mean, melt_std, melt.dtype.name} after masking')
            melt = melt.astype(np.float32) # note: src is float64, so converting to float32 will remove precision
            melt = torch.from_numpy(melt.transpose((2, 0, 1))).contiguous() # convert to tensor
            if self.cfg['normalize_melt']:
                melt = T.Normalize(mean=[self.cfg['mean_melt']], std=[self.cfg['std_melt']])(melt)
                raise NotImplementedError('need to implement that the invalid pixels in the normalized meltwater targets are set to zero again. or set normalize_melt to False.')
                # todo: implement masking of invalid pixels in torch
                melt = np.ma.array(melt, mask=mask_nans).filled(fill_value=0) # Fill masked values with zero.
            
            # Adding melt mask. Masked pixel is 1 and valid pixel is 0.
            melt_mask = mask_nans.astype(np.float32) # convert to float32, bc ML model currently only accepts float. 
            melt_mask = torch.from_numpy(melt_mask.transpose((2,0,1))).contiguous() # to torch
        else:
            # Targets are empty during deployment, because no ground-truth meltwater was observed
            melt = torch.zeros((1, height, width), dtype=self.dtype).contiguous()
            melt_mask = torch.zeros((1, height, width), dtype=self.dtype).contiguous() # All pixels valid.
            mask_nans = np.zeros((height, width, 1),dtype=bool)
        
        # Adding static landmask to mask
        if 'path_landmask' in self.cfg.keys():
            if self.cfg['path_landmask'] is not None:
                landmask_filepath = Path(self.cfg['data_root'])/Path(self.cfg['path_landmask'])
                landmask = open_cropped_tif(str(landmask_filepath), self.cfg['img_size'], offsets)
                if self.verbose:
                    print(f'landmask min, max, mean, std:\t{landmask.min(), landmask.max(), landmask.mean(), landmask.std(), landmask.dtype.name}')
                melt_mask = landmask == -1 # Ocean has label -1. Land has 1. We want to mask out the ocean. so we convert -1 -> 1 and 1 -> 0.
                land_mask = melt_mask.copy()
                melt_mask = np.ma.mask_or(mask_nans, melt_mask, shrink=False) # Create union mask of both masks        
                melt_mask = melt_mask.astype(np.float32) # creates memory overhead but that's okay for now
                melt_mask = torch.from_numpy(melt_mask.transpose((2,0,1))).contiguous() # to torch

                # Add landmask as input to ML model. Other parts of the mask cannot be added as they'd use data from the ground-truth target.
                if 'landmask' in self.cfg['in_keys_static']:
                    land_mask = land_mask.astype(np.float32)
                    land_mask = torch.from_numpy(land_mask.transpose((2,0,1))).contiguous() # to torch
                    inputs[ch_idx:ch_idx+1,...] = land_mask
                    meta['channels'][ch_idx] = 'landmask' 
                    ch_idx += 1

        # Add predictions of previous models to input. For example, to use a persistence forecast as prior to the neural network.
        if 'predictions_' in '\t'.join(self.cfg['in_keys']):
            # Get list of all model_keys if the prediction of multiple models is used as input.
            predictions_model_keys = [key for key in self.cfg['in_keys'] if 'predictions_' in key]
            for predictions_model_key in predictions_model_keys:
                predictions = open_cropped_tif(str(data_stack[predictions_model_key]), self.cfg['img_size'], offsets)
                predictions = np.ma.array(predictions, mask=np.ma.masked_invalid(predictions).mask).filled(fill_value=0) # Fill masked values with zero.
                predictions = predictions.astype(np.float32) # note: src is float64, so converting to float32 will remove precision
                predictions = torch.from_numpy(predictions.transpose((2, 0, 1))).contiguous() # convert to tensor
                if self.cfg['normalize_melt']:
                    raise NotImplementedError('normalization of meltwater predictions is not implemented yet.')
                inputs[ch_idx:ch_idx+1,...] = predictions
                meta['channels'][ch_idx] = predictions_model_key 
                ch_idx += 1

        # Add auxiliary inputs
        target_date = datetime.datetime.strptime(Path(meta['filename']).stem, "%Y_%m_%d")
        # Add time embedding. The time embedding are two images stacked onto the other inputs
        #  with the values sin(day in year) and cos(day in year)
        if 'time_sin' in self.cfg['in_keys_aux']:
            if 'time_cos' in self.cfg['in_keys_aux']:
                # Extract day between 0 and 365
                jan_1st = datetime.datetime.strptime(f'{target_date.year}_01_01', "%Y_%m_%d")
                day_in_yr = (target_date - jan_1st).days
                # Convert date to sin and cos that start on Jan 1st and end on Dec 31st
                time_sin = np.sin(float(day_in_yr)/365. * 2. * np.pi)
                time_cos = np.cos(float(day_in_yr)/365. * 2. * np.pi)
                time_sin = time_sin * torch.ones(melt.shape, dtype=self.dtype)
                time_cos = time_cos * torch.ones(melt.shape, dtype=self.dtype)
                inputs[ch_idx:ch_idx+1,...] = time_sin
                inputs[ch_idx+1:ch_idx+2,...] = time_cos
                meta['channels'][ch_idx] = 'time_sin' 
                meta['channels'][ch_idx+1] = 'time_cos' 
                ch_idx += 2
            else:
                raise ValueError('Using time embedding requires both time_sin and time_cos in in_keys_aux')
        elif 'time_cos' in self.cfg['in_keys_aux']:
            raise ValueError('Using time embedding requires both time_sin and time_cos in in_keys_aux')

        # Add month index. The month index is a single image that contains the index of month between 1 and 12 
        #  that corresponds to the target image 
        if 'time_month_idx' in self.cfg['in_keys_aux']:
            # Extract month between 1 and 12
            month_idx = target_date.month
            time_month_idx = month_idx * torch.ones(melt.shape, dtype=self.dtype)
            inputs[ch_idx:ch_idx+1,...] = time_month_idx
            meta['channels'][ch_idx] = 'time_month_idx' 
            ch_idx += 1

        # Add PMW winter means.  
        if 'pmw_winter_mean' in self.cfg['in_keys_aux']:
            year = target_date.year
            pmw_winter_mean_filepath = Path(self.cfg['data_root'])/Path(self.cfg['path_pmw_winter_mean'])
            pmw_winter_mean_filepath = pmw_winter_mean_filepath / f'{year}_jan_feb.tif'
            pmw_winter_mean = open_cropped_tif(str(pmw_winter_mean_filepath), self.cfg['img_size'], offsets)
            assert pmw_winter_mean.dtype == np.float32, f'Expected dtype float32, but got {pmw_winter_mean.dtype} for pmw_winter_mean'
            pmw_winter_mean = torch.from_numpy(pmw_winter_mean.transpose((2, 0, 1))).contiguous() # Pytorch uses channels-first: (c, h, w)
            inputs[ch_idx:ch_idx+1,...] = pmw_winter_mean
            meta['channels'][ch_idx] = 'pmw_winter_mean' 
            ch_idx += 1

        # Concatenate in- and output images
        # inputs = torch.cat((pmw, modis, mar_wa1, dem), dim=0) # dem
        targets = melt
        targets_mask = melt_mask

        return inputs, targets, targets_mask, meta

    def get_channel_idx(self, meta, channel_key):
        """
        Returns idx to channel with channel_key in inputs tensor
        Runtime complexity O(1)
        """
        # Extract index of desired channel in channel dictionary from meta information
        #  note: this is value is expected to be the same as ch_idx_in_inputs, but it 
        #  could differ if the dataset was edited after construction.
        sample_in_batch_idx = 0 # takes first image in batch and assumes that all 
            # samples in batch have the same channel order
        channels = [value[sample_in_batch_idx] for value in meta['channels'].values()]
        ch_idx_in_dict = channels.index(channel_key)
        ch_idx_in_inputs = list(meta['channels'].keys())[ch_idx_in_dict]
        return ch_idx_in_inputs

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--parallel', action='store_true', default=False, 
                        help='Enable parallel')
    parser.add_argument('--cfg_path', type=str, default='runs/unet/sample/config/config.yaml', 
                        help='Path to config yaml')
    parser.add_argument('--test_dataset', action='store_true', default=False, 
                        help='If True, initializes the dataset class with dummy parameters')
    parser.add_argument('--test_augmentations', action='store_true', default=False, 
                        help='If True, tests data augmentations')
    parser.add_argument('--create_csv_w_split_cfg', action='store_true', default=False, 
                        help='Creates a csv with split cfg that is specified in cfg_path')
    parser.add_argument('--create_csv_w_deploy_split', action='store_true', default=False, 
                        help='Creates a csv with a split for deployment')
    parser.add_argument('--split_cfg', type=str, default='stratified_time', 
                        help='Type of data split that should be craeted, e.g., day, or stratified_time')
    parser.add_argument('--check_for_invalid_entries', action='store_true', default=False, 
                        help='If True and create_csv_w_split_cfg==True, some channels will'\
                             'be checked for invalid entries. This process can take ~5min.')
    return parser.parse_args()

if __name__ == "__main__":
    """
    HRMeltDataset
    """
    gdal.UseExceptions() # Enable gdal error messages. 

    args = get_args()
    random.seed(0)
    torch.manual_seed(0)

    if args.create_csv_w_deploy_split:
        # Creates a deploy.csv files that contain the
        #  filenames of each image in each data split
        cfg = yaml.safe_load(open(args.cfg_path, 'r'))
        cfg['check_for_invalid_entries'] = True
        if 'time_interpolate_sar' in cfg['in_keys']:
            cfg['in_keys'].remove('time_interpolate_sar')

        cfg['split_cfg'] = None # switch off loading filepaths from .csv files 
        dataset = HRMeltDataset(cfg=cfg, split='deploy', 
                                verbose=True, check_data_in_parallel=args.parallel)

        # Extract list of filenames from dataset
        filenames = []
        for data_stack in dataset.data:
            for i, in_key in enumerate(cfg['in_keys']):
                if i == 0:
                    filenames.append(data_stack[in_key].name)
                elif filenames[-1] != data_stack[in_key].name:
                    print(f'Warning dates dont match in {data_stack}')

        # Store list of filenames as a csv using pandas
        series = pd.Series(filenames)
        Path(cfg['path_deploy_split_csv']).parent.mkdir(parents=True, exist_ok=True) # Create parent directory, if not exist
        print('Saving .csv at: ', cfg['path_deploy_split_csv'])
        series.to_csv(cfg['path_deploy_split_csv'], header=False, index=False)

    if args.create_csv_w_split_cfg:
        # Creates the train, val, test .csv files that contain the
        #  filenames of each image in each data split
        cfg = yaml.safe_load(open(args.cfg_path, 'r'))
        cfg['create_csv_w_split_cfg'] = True
        if args.check_for_invalid_entries:
            cfg['check_for_invalid_entries'] = True
        if 'time_interpolate_sar' in cfg['in_keys']:
            # remove time_interpolate_sar from inputs, because dataset 
            #  will only grab filestacks with every channel available 
            #  and time_interpolate_sar input might not have been created 
            #  yet.
            cfg['in_keys'].remove('time_interpolate_sar')

        dataset = HRMeltDataset(cfg=cfg, verbose=True)
        dataset.verbose = False
        _, _, _ = dataset.create_data_splits(
            args.split_cfg,
            val_percent=cfg['val_percent'],
            test_percent=cfg['test_percent'],
            seed=cfg['seed'],
            create_csv_w_split_cfg=cfg['create_csv_w_split_cfg'],
            path_train_split_csv=cfg['path_train_split_csv'],
            path_val_split_csv=cfg['path_val_split_csv'],
            path_test_split_csv=cfg['path_test_split_csv'],
            path_all_split_csv=cfg['path_all_split_csv'],
            n_imgs_per_month=cfg['n_imgs_per_month']
            )

    if args.test_dataset:
        cfg = yaml.safe_load(open('runs/unet/data_s1x/config/config.yaml', 'r'))
        cfg['split_cfg'] = 'csv'
        cfg['create_csv_w_split_cfg'] = False

        dataset = HRMeltDataset(cfg, verbose=True)
        print('Loaded dataset with len: ',len(dataset))

    if args.test_augmentations:
        columns = 4
        fig, axs = plt.subplots(1, columns, figsize=(10,4))
        im_idx = 0

        for i, pair in enumerate([[1., 1.],[29., 9.],[29.,15.],[45., 15.]]):
            kernel, sigma = pair 

            dataset.cfg['pmw_GaussianBlur_kernel_size'] = kernel
            dataset.cfg['pmw_GaussianBlur_sigma'] = sigma

            np.random.seed(cfg['seed'])
            random.seed(cfg['seed'])
            torch.manual_seed(cfg['seed'])

            inputs, _, _, _ = dataset.__getitem__(idx=im_idx)
            pmw = inputs[0]

            plt.subplot(1,columns,i+1)
            ax = plt.imshow(pmw.cpu().numpy())
            plt.colorbar(ax, orientation='horizontal', fraction=0.05, pad=0.01)
            plt.title(f'kernel: {dataset.cfg["pmw_GaussianBlur_kernel_size"]}, sigma {dataset.cfg["pmw_GaussianBlur_sigma"]}')
        plt.savefig(f"references/figures/tmp/pmwaugmented.png")

