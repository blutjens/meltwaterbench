"""
Plots MAE as a function of the length of the observational gap, i.e., the
number of days until the closest preceding or succeeding valid observation.
For every test image we build a per-pixel count of days until the preceding
(and succeeding) train observation, bin the absolute prediction errors by that
count, and plot MAE over gap length. Preceding gaps are plotted at negative and
succeeding gaps at positive x-values, so each MAE bin appears twice.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from osgeo import gdal
import matplotlib.pyplot as plt
from tqdm import tqdm

from hrmelt.dataset import open_cropped_tif

def get_melt_mask(filename, path_melt, img_size, landmask):
    """Returns melt_mask (1=invalid) for a melt target, combining nan and landmask."""
    melt = open_cropped_tif(str(Path(path_melt)/filename), img_size, offsets=(0, 0))
    mask_nans = np.ma.masked_invalid(melt).mask # Mask nans from, e.g., overexposure
    melt_mask = np.ma.mask_or(mask_nans, landmask, shrink=False) # Union of nan and ocean mask
    return melt_mask[..., 0] # (h, w)

if __name__ == '__main__':
    gdal.UseExceptions()

    # Paths
    path_predictions = '/Users/bjoern/data/hrmelt/meltwaterbench/interim/runs/unet_smp/data_v1_4/deploy/'
    path_landmask = '/Users/bjoern/data/hrmelt/meltwaterbench/raw/Helheim_data/reprojected_100m/GIMP_Mask/landMask_100m.tif'
    path_melt = '/Users/bjoern/data/hrmelt/meltwaterbench/raw/Helheim_data/reprojected_100m/SAR/S1Xv1.2/HH_PercentMelt/'
    path_test_csv = './runs/unet_smp/data_v1_4/config/test.csv'
    path_train_csv = './runs/unet_smp/data_v1_4/config/train.csv'
    path_figure = './references/figures/benchmark/unet_smp/data_v1_4/mae_over_observation_gap.png'
    years = range(2018, 2024) # Consider only 2018 to 2023

    # Reference size from landmask
    landmask_full = open_cropped_tif(path_landmask, None, None)
    img_size = landmask_full.shape[:2]
    landmask = (landmask_full == -1) # Ocean has label -1, land has 1. Mask out ocean.

    # Read split filenames
    test_files = pd.read_csv(path_test_csv, header=None)[0].tolist()
    train_files = pd.read_csv(path_train_csv, header=None)[0].tolist()
    train_dates = sorted(datetime.strptime(Path(f).stem, '%Y_%m_%d') for f in train_files)
    test_dates = sorted(datetime.strptime(Path(f).stem, '%Y_%m_%d') for f in test_files)
    test_dates = [d for d in test_dates if d.year in years]
    train_dates_set = set(train_dates)

    # Accumulate absolute errors, squared errors, and pixel counts per gap length (in days)
    abs_err_prec, sq_err_prec, count_prec = {}, {}, {} # preceding: {n_days: sum|abs_err|, sum abs_err^2, n_pixels}
    abs_err_succ, sq_err_succ, count_succ = {}, {}, {} # succeeding

    for test_date in tqdm(test_dates):
        filename = test_date.strftime('%Y_%m_%d') + '.tif'
        if not (Path(path_predictions)/filename).exists(): # Skip dates without a prediction tif
            continue
        # Prediction (deploy) and target (melt), masked to valid land+target pixels
        pred = open_cropped_tif(str(Path(path_predictions)/filename), img_size, (0, 0))[..., 0]
        melt = open_cropped_tif(str(Path(path_melt)/filename), img_size, (0, 0))
        mask_nans = np.ma.masked_invalid(melt).mask
        melt = np.ma.array(melt, mask=mask_nans).filled(fill_value=0)[..., 0].astype(np.float32)
        melt_mask = np.ma.mask_or(mask_nans, landmask, shrink=False)[..., 0] # 1=invalid
        abs_err = np.abs(pred - melt)

        # num_days_until_preceding_obs: init 0, valid only where test valid (melt_mask==0)
        num_prec = np.zeros(img_size, dtype=np.int32)
        prec_invalid = melt_mask.copy() # invalid where test invalid or no preceding obs found
        found_prec = np.zeros(img_size, dtype=bool)
        day_delta = 1
        while (test_date - timedelta(days=day_delta)).year == test_date.year:
            prev = test_date - timedelta(days=day_delta)
            if prev in train_dates_set:
                prev_mask = get_melt_mask(prev.strftime('%Y_%m_%d') + '.tif', path_melt, img_size, landmask)
                newly = (~found_prec) & (~melt_mask) & (~prev_mask) # observed now, not before
                num_prec[newly] = day_delta
                found_prec |= newly
            day_delta += 1
        prec_invalid |= ~found_prec # no preceding observation this year

        # num_days_until_succeeding_obs
        num_succ = np.zeros(img_size, dtype=np.int32)
        succ_invalid = melt_mask.copy()
        found_succ = np.zeros(img_size, dtype=bool)
        day_delta = 1
        while (test_date + timedelta(days=day_delta)).year == test_date.year:
            nxt = test_date + timedelta(days=day_delta)
            if nxt in train_dates_set:
                nxt_mask = get_melt_mask(nxt.strftime('%Y_%m_%d') + '.tif', path_melt, img_size, landmask)
                newly = (~found_succ) & (~melt_mask) & (~nxt_mask)
                num_succ[newly] = day_delta
                found_succ |= newly
            day_delta += 1
        succ_invalid |= ~found_succ

        # Bin absolute errors by gap length
        for n_days in np.unique(num_prec[~prec_invalid]):
            sel = (num_prec == n_days) & (~prec_invalid)
            abs_err_prec[n_days] = abs_err_prec.get(n_days, 0.) + abs_err[sel].sum()
            sq_err_prec[n_days] = sq_err_prec.get(n_days, 0.) + (abs_err[sel]**2).sum()
            count_prec[n_days] = count_prec.get(n_days, 0) + sel.sum()
        for n_days in np.unique(num_succ[~succ_invalid]):
            sel = (num_succ == n_days) & (~succ_invalid)
            abs_err_succ[n_days] = abs_err_succ.get(n_days, 0.) + abs_err[sel].sum()
            sq_err_succ[n_days] = sq_err_succ.get(n_days, 0.) + (abs_err[sel]**2).sum()
            count_succ[n_days] = count_succ.get(n_days, 0) + sel.sum()

    # Aggregate gap lengths into bins 1, 2, 3-4, 5-9, 10+
    bins = [(1, 1), (2, 3), (4, 6), (6, np.inf)]
    bin_labels = ['1', '2-3', '4-6', '6+']
    def bin_mae(abs_err, sq_err, count):
        mae, std = [], []
        for lo, hi in bins:
            err = sum(v for n, v in abs_err.items() if lo <= n <= hi)
            sq = sum(v for n, v in sq_err.items() if lo <= n <= hi)
            cnt = sum(v for n, v in count.items() if lo <= n <= hi)
            mean = err / cnt if cnt > 0 else np.nan
            mae.append(mean)
            std.append(np.sqrt(max(sq / cnt - mean**2, 0.)) if cnt > 0 else np.nan) # std of per-pixel abs error
        return np.array(mae), np.array(std)
    mae_prec, std_prec = bin_mae(abs_err_prec, sq_err_prec, count_prec)
    mae_succ, std_succ = bin_mae(abs_err_succ, sq_err_succ, count_succ)

    var_prec = std_prec**2
    var_succ = std_succ**2

    # Plot: preceding at negative x, succeeding at positive x
    x = np.arange(len(bins)) + 1
    fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
    ax.plot(-x, mae_prec, marker='o', color='tab:blue', label='preceding')
    # ax.fill_between(-x, mae_prec - var_prec, mae_prec + var_prec, color='tab:blue', alpha=0.2)
    ax.plot(x, mae_succ, marker='o', color='tab:orange', label='succeeding')
    # ax.fill_between(x, mae_succ - var_succ, mae_succ + var_succ, color='tab:orange', alpha=0.2)
    ax.axvline(0, color='gray', linewidth=0.8)
    ax.set_xticks(np.concatenate((-x, x)))
    ax.set_xticklabels(['-' + l for l in bin_labels] + bin_labels)
    ax.set_xlabel('Days to closest observation')
    ax.set_ylabel('MAE')
    ax.legend()
    plt.tight_layout()

    Path(path_figure).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_figure)
    plt.close()
    print(f'Saved figure to {path_figure}')
