"""
Applies a trained random forest baseline to a single full-scale image
and saves the prediction as .tif and .png. Every pixel is used as a
feature vector, regardless of the target mask. Ocean pixels are set to
zero using the dataset landmask.
"""
import yaml
import argparse
import logging
import joblib
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from osgeo import gdal # rasterio in dataloader uses gdal. Import to suppress warning msg.
import torch
from torchvision.utils import save_image

from hrmelt.dataset import HRMeltDataset
from hrmelt.dataset import open_cropped_tif
from hrmelt.utils.utils import save_tensor_as_tif
from hrmelt.utils.utils import set_all_seeds
from hrmelt.utils.utils import get_size_of_tif

def predict(cfg, rf_best_model, data_split, verbose=False):
    """
    Predicts full-scale images with the trained random forest and saves
    each as .tif and .png. Uses all pixels as features and zeros out ocean
    pixels via the landmask.
    """
    num_in_channels = cfg['in_channels']
    max_num_test_samples = cfg['max_num_test_samples']

    # Create predictions over full image
    if 'path_melt_reference' in cfg:
        tif_size = get_size_of_tif(str(Path(cfg['data_root'])/Path(cfg['path_melt_reference'])))
    else:
        print('INFO: Reference target not found. Using hardcoded tif size.')
        tif_size = [2863, 1633]
    cfg['img_size'] = tif_size

    # Initialize test dataset, similar to hrmelt/predict.py
    assert cfg['split_cfg'] == 'csv', 'random_forest currently only accepts image paths in .csv format'
    print('Loading test dataset...')
    dataset = HRMeltDataset(cfg=cfg, split=data_split, verbose=False)
    dataset.sort()
    print('Loaded test dataset.')

    # Load the landmask once; ocean has label -1, land has 1
    landmask_filepath = Path(cfg['data_root']) / Path(cfg['path_landmask'])
    landmask = open_cropped_tif(str(landmask_filepath), cfg['img_size'], [0, 0])
    ocean = landmask == -1
    ocean = torch.from_numpy(ocean.astype(np.float32).transpose((2, 0, 1)))

    # Process each sample
    n_imgs = len(dataset) if max_num_test_samples is None else min(max_num_test_samples, len(dataset))
    pred_paths = []
    for idx in tqdm(range(n_imgs), unit='img'):
        inputs, _, _, meta = dataset.__getitem__(idx)
        height, width = inputs.shape[-2:]

        # Flatten sample into (num_pixels, num_in_channels); discard targets
        X_test = inputs.numpy().reshape(num_in_channels, -1).T # (n_px, n_ch)

        # Query predictions and reshape into an image
        predictions = rf_best_model.predict(X_test)
        pred = torch.from_numpy(predictions.astype(np.float32)).reshape(1, height, width)

        # Apply landmask to fill zero for all predictions over the ocean
        if cfg['apply_landmask_to_predictions']:
            pred = torch.mul(pred, (1 - ocean))

        # Save prediction as tif and png, similar to hrmelt/predict.py
        tif_path = Path(meta['path_melt'])
        new_tif_path = Path(cfg['path_predictions']) / Path(meta['filename'])
        save_tensor_as_tif(pred, tif_path=str(tif_path), new_tif_path=str(new_tif_path), verbose=verbose)
        new_png_path = Path(new_tif_path).with_suffix('.png')
        save_image(pred, str(new_png_path))
        if verbose:
            logging.info(f'Saved: {new_png_path}')

        # Plot the image
        plt.figure()
        plt.imshow(pred.squeeze(0).numpy())
        plt.colorbar(orientation='horizontal', fraction=0.05, pad=0.01)
        plt.title(Path(meta['filename']).stem)
        plt.savefig(Path(new_tif_path).with_suffix('.plot.png'))
        plt.close()

        pred_paths.append(new_tif_path)

    return pred_paths

def get_args():
    parser = argparse.ArgumentParser(description='Create random forest predictions')
    parser.add_argument('--load', '-f', type=str, default=None, help='Overwrite path to checkpoint')
    parser.add_argument('--cfg_path', type=str, default='runs/random_forest/data_v1_4/config/config.yaml',
                        help='Path to config yaml')
    parser.add_argument('--parallel', action='store_true', default=False, help='Enable parallel inference')
    parser.add_argument('--data_split', type=str, default='test', help='Split [train, val, or test] for which the'\
                         'predictions will be calculated')
    parser.add_argument('--verbose', action='store_true', default=False, help='Print verbose logs')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    gdal.UseExceptions() # Enable gdal error messages (otherwise a warning is raised)

    cfg = yaml.safe_load(open(args.cfg_path, 'r'))
    cfg['num_workers'] = cfg['num_workers'] if args.parallel else 1 # n_jobs is 1 unless --parallel
    cfg['in_channels'] = len(cfg['in_keys']) + len(cfg['in_keys_static']) + len(cfg['in_keys_aux'])
    set_all_seeds(cfg['seed'], use_deterministic_algorithms=cfg['use_deterministic_algorithms'])

    # Load the trained random forest
    if args.load:
        checkpoint_path = args.load
    else:
        checkpoint_path = Path(cfg['path_checkpoints']) / 'checkpoint.joblib'
    rf_best_model = joblib.load(str(checkpoint_path))
    rf_best_model.n_jobs = cfg['num_workers'] # Parallelize inference across cores
    logging.info(f'Model loaded from {checkpoint_path}')

    # Count number of split thresholds
    print(f'Number of split thresholds: {sum((t.tree_.children_left != -1).sum() for t in rf_best_model.estimators_)}')

    predict(cfg,
            rf_best_model,
            data_split=args.data_split,
            verbose=args.verbose)
