"""
Trains a baseline random forest regressor to predict surface meltwater
fraction from the input channels. A pixel is one sample; the forest is
fit on a random subset of valid pixels and tuned with a randomized
hyperparameter search over a predefined train/val split.
"""
import yaml
import argparse
import logging
import joblib
import numpy as np
from pathlib import Path
from tqdm import tqdm
from osgeo import gdal # rasterio in dataloader uses gdal. Import to suppress warning msg.
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import PredefinedSplit

from hrmelt.dataset import HRMeltDataset
from hrmelt.utils.utils import set_all_seeds

def sample_pixels(dataset, num_in_channels, subsampled_px_ratio, max_samples=None):
    """
    Iterates over dataset and randomly subsamples valid pixels without
    replacement. Each sample returns inputs (n_ch, h, w), a target
    (1, h, w), and a target_mask (1, h, w) with 1. for invalid pixels.
    Args:
        max_samples int: If specified, only iterate over the first max_samples images.
    Returns:
        X np.ndarray(n_samples, num_in_channels), y np.ndarray(n_samples)
    """
    X = np.empty((0, num_in_channels), dtype=np.float32)
    y = np.empty((0), dtype=np.float32)
    n_imgs = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    for idx in tqdm(range(n_imgs), unit='img'):
        inputs, targets, targets_mask, _ = dataset.__getitem__(idx)
        inputs = inputs.numpy().reshape(num_in_channels, -1).T # (n_px, n_ch)
        targets = targets.numpy().reshape(-1) # (n_px)
        valid = targets_mask.numpy().reshape(-1) == 0. # mask is 1. for invalid px
        valid_idx = np.flatnonzero(valid)
        n_sample = int(valid_idx.size * subsampled_px_ratio)
        sample_idx = np.random.choice(valid_idx, size=n_sample, replace=False)
        X = np.vstack((X, inputs[sample_idx]))
        y = np.hstack((y, targets[sample_idx]))
    return X, y

def train(cfg):
    num_in_channels = cfg['in_channels']
    subsampled_px_ratio = cfg['subsampled_px_ratio']

    # Initialize train and val datasets, similar to hrmelt/train.py
    assert cfg['split_cfg'] == 'csv', 'random_forest currently only accepts image paths in .csv format'
    print('Loading train and val datasets...')
    train_set = HRMeltDataset(cfg=cfg, split='train', verbose=False)
    val_set = HRMeltDataset(cfg=cfg, split='val', verbose=False)

    # Subsample valid pixels into feature/target arrays
    print('Sampling pixels from train set...')
    X_train, y_train = sample_pixels(train_set, num_in_channels, subsampled_px_ratio, cfg['max_num_train_samples'])
    print('Sampling pixels from val set...')
    X_val, y_val = sample_pixels(val_set, num_in_channels, subsampled_px_ratio, cfg['max_num_val_samples'])

    # Define train and validation data split (-1 = train, 0 = val fold)
    X_train_val = np.vstack((X_train, X_val))
    y_train_val = np.hstack((y_train, y_val))
    num_train_samples = X_train.shape[0]
    num_val_samples = X_val.shape[0]
    train_val_fold = np.zeros(num_train_samples + num_val_samples)
    train_val_fold[:num_train_samples] = -1
    train_val_fold[num_train_samples:] = 0
    train_val_split = PredefinedSplit(test_fold=train_val_fold)

    class VerboseRandomForestRegressor(RandomForestRegressor):
        """Print hyperparameter before fitting"""
        def fit(self, X, y, **kwargs):
            params = self.get_params()
            print(f"\n[Fitting candidate] {params}", flush=True)
            return super().fit(X, y, **kwargs)

    rf_model = VerboseRandomForestRegressor(n_jobs=cfg['num_workers'], random_state=cfg['seed'])

    rf_param_dist = {
        "n_estimators": [200], # np.linspace(start=50, stop=500, num=10).astype(int),
        "max_features": ['log2'], # ['log2', None],
        "max_depth": [16],
        "max_samples": [0.05], # [0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
        "min_samples_split": [4],
        "min_samples_leaf": [4],
        "bootstrap": [True],
        "criterion": ["squared_error"], # "absolute_error"],#, "squared_error"], 
    }

    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=rf_param_dist,
        n_iter=cfg['num_hyperparam_tries'],
        cv=train_val_split,
        scoring="neg_mean_absolute_error", # L1 loss
        n_jobs=1, # Each RF is parallelized
        random_state=cfg['seed'],
        verbose=3
    )

    random_search.fit(X_train_val, y_train_val)

    # Print best parameter choices and the associated validation score
    print('Best parameters: ', random_search.best_params_)
    print('Best validation neg_mean_absolute_error: ', random_search.best_score_)

    rf_best_model = random_search.best_estimator_
    rf_best_model.__class__ = RandomForestRegressor

    # Save the trained model to cfg['path_checkpoints']
    Path(cfg['path_checkpoints']).mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(cfg['path_checkpoints']) / 'checkpoint.joblib'
    joblib.dump(rf_best_model, str(checkpoint_path))
    logging.info(f'Checkpoint saved to {checkpoint_path}')

    return rf_best_model

def get_args():
    parser = argparse.ArgumentParser(description='Train the random forest baseline')
    parser.add_argument('--cfg_path', type=str, default='runs/random_forest/data_v1_4/config/config.yaml',
                        help='Path to config yaml')
    parser.add_argument('--verbose', action='store_true', default=False, help='Print verbose logs')
    parser.add_argument('--parallel', action='store_true', default=False, help='Enable parallel training')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    gdal.UseExceptions() # Enable gdal error messages (otherwise a warning is raised)

    cfg = yaml.safe_load(open(args.cfg_path, 'r'))
    cfg['num_workers'] = cfg['num_workers'] if args.parallel else 1 # n_jobs is 1 unless --parallel
    cfg['in_channels'] = len(cfg['in_keys']) + len(cfg['in_keys_static']) + len(cfg['in_keys_aux'])
    set_all_seeds(cfg['seed'], use_deterministic_algorithms=cfg['use_deterministic_algorithms'])

    train(cfg)
