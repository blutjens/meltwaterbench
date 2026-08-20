# MeltwaterBench

Official repository for the paper 'MeltwaterBench: Deep learning for spatiotemporal downscaling of surface meltwater'. We set-up a benchmark dataset for spatiotemporally downscaling a regional climate model, passive microwave, and digital elevation model onto high-resolution synthetic aperture radar-derived observations of surface meltwater, as shown in Fig. 1. This code contains all steps necessary to reproduce the results in the paper, including the data download, fitting baselines, and running evaluation. 

- To reproduce the paper follow the instructions on [installation](#Installation), [data download](#download-dataset), and [model training](#Train-the-more-complex-UNet)
- To add a new model, please see the instruction in [adding a new model](#Adding-a-new-model)
- To get more information see the [preprint](https://arxiv.org/abs/2512.12142) or a video of the [generated 100m product](https://www.youtube.com/watch?v=OaonUT6dIbg).

Fig. 1, MeltwaterBench downscaling task
<img width="1255" height="540" alt="image" src="https://github.com/user-attachments/assets/41132419-6be3-4087-b905-1ee56e7aae6e" />

## Installation
We recommend installing the project via [conda](https://docs.conda.io/en/latest/).
```
git clone git@github.com:blutjens/meltwaterbench.git
cd meltwaterbench
conda env create -f environment.yml
conda activate hrmelt
pip install -e .
```

If install crashes try `pip install --find-links=https://girder.github.io/large_image_wheels --no-cache GDAL`
 or see [debugging](#debugging) section.

## Getting started
Run the following scripts to verify that the repo works on a small sample dataset:

#### Explore the dataset
```
jupyter notebook notebooks/explore_data.ipynb
# For vscode: $ python -m ipykernel install --user --name=hrmelt
```

#### Overfit vanilla unet on sample data
```
python hrmelt/train.py --no_wandb --parallel --cfg_path=runs/unet/sample/config/config.yaml
# If process run out of memory: try --exclude_ssim
```

#### Run inference on sample data
This creates large tifs by convoluting the tile-predicting model across the image.
```
python hrmelt/predict.py --parallel --cfg_path runs/unet/sample/config/config.yaml --load runs/unet/sample/checkpoints/checkpoint_epoch5.pth
```

## Download dataset
#### Preferred option via git
```
# Make sure you have git-lfs installed (https://git-lfs.github.com/)
git lfs install # Init lfs
# Download the core dataset and created product (~100GB)
git clone https://huggingface.co/datasets/blutjens/meltwaterbench --branch main --single-branch
# Test by looking at the created 100m daily product at:
./interim/runs/unet_smp/data_v1_4/deploy
```
#### Optional data instructions
Optional: Download dataset via huggingface_hub. Avoids git lfs, but may be rate limited.
```
# Install huggingface_hub in a virtual environemnt (e.g., conda activate hrmelt)
pip install huggingface_hub
# Open a Python command line interface
python
# Run this Python script to download the core dataset and created product (~100GB)
from huggingface_hub import snapshot_download
snapshot_download(
  repo_id="blutjens/meltwaterbench",
  repo_type='dataset',
  revision="main",
  cache_dir="./" # Path to store dataset
)
```
Optional: Download via git with ssh access
```
# Create a huggingface account (https://huggingface.co/join)
# Add the public ssh key you're using for git to your huggingface profile (https://huggingface.co/docs/hub/security-git-ssh). You may find it with `cat ~/.ssh/id_ed25519.pub`
ssh -T git@hf.co # Test ssh connection
git clone git@hf.co:datasets/blutjens/meltwaterbench --branch main --single-branch
```
Optional: Download the auxiliary dataset (~250GB) with:
```
git clone git@hf.co:datasets/blutjens/meltwaterbench --branch S1Xv1.2_aux --single-branch
```


## Reproduce paper results
#### Setup data config
```
Open the config at `runs/unet_smp/data_v1_4/config/config.yaml` and set `data_root: /path/to/store/dataset`. Repeat process for every baseline model you're reproducing.
```

#### Train the more complex UNet
This is training the configuration that's reported in the paper.
```
python hrmelt/train.py --no_wandb --parallel --cfg_path 'runs/unet_smp/data_v1_4/config/config.yaml'
```

#### Run inference on the validation set
This creates and stores predictions into cfg['path_predictions']. 
```
python hrmelt/predict.py --parallel --cfg_path='runs/unet_smp/data_v1_4/config/config.yaml' --load='runs/unet_smp/data_v1_4/checkpoints/checkpoint_epoch1025.pth' --data_split test
```

#### Run inference with all baseline models
This will create and store predictions for every baseline model. Metrics will be computed next. To train the baseline models see [fitting baselines](#fit-the-baseline-models).
```
# Time interpolate SAR model. Forecasts average of previous n melt images.
python hrmelt/models/time_interpolate_sar/predict.py --cfg_path runs/time_interpolate_sar/data_v1_4/config/config.yaml --target_split test

# Interpolate MAR model. Interpolates liquid water content in MAR over space. (interpolate_mar)
python hrmelt/models/interpolate_mar/predict.py --cfg_path runs/interpolate_mar/data_v1_4/config/config.yaml --data_split test

# Threshold PMW model
python hrmelt/models/threshold_pmw/predict.py --cfg_path runs/threshold_pmw/data_v1_4/config/config.yaml --data_split test

# Linear model wrt. digital elevation (linear_dem)
python hrmelt/predict.py --parallel --cfg_path runs/linear_dem/data_v1_4/config/config.yaml --load='runs/linear_dem/data_v1_4/sweep/task-8/checkpoints/checkpoint_epoch101.pth' --data_split test

# Random forest
python hrmelt/models/random_forest/predict.py --parallel --cfg_path=runs/random_forest/data_v1_4/config/config.yaml --load=/home/gridsan/lutjens/hrmelt/runs/random_forest/data_v1_4/checkpoints/ratio_020_mse_v05/checkpoint.joblib --data_split test

# Create Deeplabv3 predictions
python hrmelt/predict.py --parallel --cfg_path='runs/deeplabv3/data_v1_4/config/config.yaml' --load='runs/deeplabv3/data_v1_4/sweep/task-7/checkpoints/checkpoint_epoch991.pth' --data_split test

# Create unet predictions
python hrmelt/predict.py --parallel --cfg_path='runs/unet_smp/data_v1_4/config/config.yaml' --load='runs/unet_smp/data_v1_4/sweep/task-1/checkpoints/checkpoint_epoch1025.pth' --data_split test

# Create vanilla unet predictions
python hrmelt/predict.py --parallel --cfg_path='runs/unet/data_v1_4/config/config.yaml' --load='runs/unet/data_v1_4/sweep/task-18/checkpoints/checkpoint_epoch451.pth' --data_split test
```

#### Benchmark: Compute evaluation metrics for every model
```
python hrmelt/eval/benchmark.py --compute_metrics --batch_size 10 --data_split test --unet_smp --linear_dem --time_interpolate_sar --interpolate_mar --deeplabv3 --threshold_pmw --random_forest
```

#### Benchmark: Plot model prediction vs. target for every validaton image
```
python hrmelt/eval/benchmark.py --plot_predictions_vs_targets --data_split test --unet_smp --linear_dem --time_interpolate_sar --interpolate_mar --threshold_pmw
```

#### Benchmark: Plot model error (prediction-target) vs. target
```
python hrmelt/eval/benchmark.py --plot_errors_vs_targets --data_split test --unet_smp --linear_dem --time_interpolate_sar --interpolate_mar  --threshold_pmw
```

#### Benchmark: Plot predicted meltwater extent over time
```
python hrmelt/eval/benchmark.py --plot_meltwater_extent_over_time --data_split test --unet_smp --time_interpolate_sar --interpolate_mar --linear_dem --threshold_pmw
python hrmelt/eval/benchmark.py --plot_meltwater_extent_over_time --data_split test --unet_smp --deeplabv3 --random_forest
```

#### Deployment: Create daily 100m predictions over 2017-23
Generate time_interpolate_sar for every day (might take ~20min)
```
python hrmelt/models/time_interpolate_sar/predict.py --cfg_path runs/time_interpolate_sar/data_v1_4/config/config.yaml --target_split deploy --save_png
```

Generate unet predictions
```
sbatch deploy.sh
python hrmelt/predict.py --parallel --cfg_path='runs/unet_smp/data_v1_4/config/config.yaml' --load='runs/unet_smp/data_v1_4/checkpoints/checkpoint_epoch1025.pth' --data_split deploy --path_time_interpolate_sar='interim/runs/time_interpolate_sar/data_v1_4/deploy/' --apply_landmask_to_predictions True
```

Generate deeplabv3+, vanilla unet, MAR, PMW, and DEM predictions
```
python hrmelt/predict.py --parallel --cfg_path='runs/deeplabv3/data_v1_4/config/config.yaml' --load='runs/deeplabv3/data_v1_4/sweep/task-7/checkpoints/checkpoint_epoch991.pth' --data_split deploy --path_time_interpolate_sar='interim/runs/time_interpolate_sar/data_v1_4/deploy/' --apply_landmask_to_predictions True
python hrmelt/predict.py --parallel --cfg_path='runs/unet/data_v1_4/config/config.yaml' --load='runs/unet/data_v1_4/sweep/task-18/checkpoints/checkpoint_epoch451.pth' --data_split deploy --path_time_interpolate_sar='interim/runs/time_interpolate_sar/data_v1_4/deploy/' --apply_landmask_to_predictions True
python hrmelt/models/interpolate_mar/predict.py --cfg_path runs/interpolate_mar/data_v1_4/config/config.yaml --data_split deploy --save_png
python hrmelt/models/threshold_pmw/predict.py --cfg_path runs/threshold_pmw/data_v1_4/config/config.yaml --data_split deploy --save_png --verbose
python hrmelt/predict.py --parallel --cfg_path runs/linear_dem/data_v1_4/config/config.yaml --load='runs/linear_dem/data_v1_4/sweep/task-8/checkpoints/checkpoint_epoch101.pth' --data_split deploy --apply_landmask_to_predictions True
```

#### Benchmark: Plot total inferred surface meltwater over study area per day
```
python hrmelt/eval/deploy.py --unet_smp --time_interpolate_sar --threshold_pmw --interpolate_mar # --deeplabv3 
```

#### Benchmark: Measure the test set variance
```
python hrmelt/eval/data_variance.py --verbose
```

#### Benchmark: Compare to station observations
```
notebooks/mevaluate_on_station_data.ipynb
```

#### Benchmark: Evaluate error over observational gap length
```
python notebooks/scripts/accuracy_over_observation_gap.py
```

#### Benchmark: Perform feature importance analysis
```
# Retrain the UNet SMP multiple times using the configs in data_v1_4_sensitivity. Afterwards call predict.py and benchmark.py on each model.
python hrmelt/train.py --no_wandb --parallel --cfg_path 'runs/unet_smp/data_v1_4_sensitivity/config/config_only_sar.yaml'
python hrmelt/train.py --no_wandb --parallel --cfg_path 'runs/unet_smp/data_v1_4_sensitivity/config/config_only_mar.yaml'
...
```

## Fit the baseline models
#### Fit baseline: interpolate_mar
```
python hrmelt/models/interpolate_mar/sweep.py --cfg_path runs/interpolate_mar/data_v1_4/config/config.yaml --sweep_path runs/interpolate_mar/data_v1_4/config/sweep.yaml
# Then copy the best parameters from the command line to the config.yaml
```

#### Fit baseline: linear_dem
```
# First, test if model can be trained
python hrmelt/train.py --no_wandb --cfg_path runs/linear_dem/data_v1_4/config/config.yaml --parallel
# Second, edit train_linear_dem.sh and kick-off hyperparam sweep
sbatch train_linear_dem.sh
# Finally copy the best parameters from wandb into the config.yaml
```

#### Fit baseline: random_forest
```
python hrmelt/models/random_forest/model.py --parallel --cfg_path=runs/random_forest/data_v1_4/config/config.yaml
# This will save out the best checkpoint
```

#### Fit baseline: deeplabv3+
```
# First, test if the model can be trained without hyperparameter tuning and wandb syncing works:
#  Requires internet access to download pretrained model weights
python hrmelt/train.py --parallel --cfg_path='runs/deeplabv3/data_v1_4/config/config.yaml' 
wandb sync --include-offline latest-run
wandb sync --clean --no-include-synced --include-offline runs/deeplabv3/data_v1_4/wandb/offline* 
# Second, test if the train.py works in sweep configuration
python hrmelt/train.py --parallel --sweep --cfg_path='runs/deeplabv3/data_v1_4/config/config.yaml' --task_id 1  --num_tasks 1
# Lastly, kick-off the hyperparameter sweep:
sbatch train.sh
conda activate hrmelt
bash hrmelt/utils/sync_wandb.sh
tail -f runs/deeplabv3/data_v1_4/sweep/task-1/task.sh.log
wandb sync --clean --include-offline runs/deeplabv3/data_v1_4/sweep/task-*/wandb/offline* # --no-include-synced
# Finally copy the best parameters into the config.yaml
```

## Adding a new model
#### Add model
To add a new model, we recommend the following steps:
- Fork the repository, clone locally, and create a new branch called <model_name>. If you're new to git follow [git workflow](#git-workflow)
- Follow the above instructions until you have the baseline machine learning model trained and evaluated
- Create a directory 'runs/model_name/experiment_name/config/' to store all new config files
- Add new model code for model architecture, train, and predict at 'hrmelt/models/model_name/'
- Optionally, adapt the dataloader in 'hrmelt/dataset.py' to the needs of your new model, e.g., by writing a childclass that inherits from HRMeltDataset

#### git workflow
```
# open a browser window
# log into your github on (https://github.com)
# open the repository at (https://github.com/blutjens/hrmelt)
# fork the repository's main branch 
# make sure ssh is set-up on your computer (https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account?platform=mac&tool=webui#adding-a-new-ssh-key-to-your-account)
# open a terminal and run:
git clone git@github.com:<your_git_username>/hrmelt.git # clone your fork
git checkout -b dev_username # create a new branch
# create the changes you like
git add changed_files.py # let the local git config know which files have been changed
git commit -m "write here what you've changed"
git push # upload the changes to the web
```

## Optional
#### Optional: Plot the dataset
```
# Plot the dataset, including all input channels, targets, and targets_mask
python hrmelt/utils/plotting.py --plot_dataset --cfg_path='runs/unet/data_v1_4/config/config.yaml'
```

# Debugging
Here is some common debugging help:

#### Debug installation
no module named 'osgeo'
```
# There was likely an issue with the gdal installation. Try install with conda:
conda install -c conda-forge gdal
# Otherwise please consult with a search engine.
```

if import torch reports OMP: Error #15, try reinstalling all with mamba:
```
conda create --name hrmelt2 python==3.9
conda activate hrmelt2
mamba install pip matplotlib scikit-image scikit-learn numpy pandas jupyter tqdm torchvision rasterio ipyleaflet localtileserver fiona gdal ray-default pytorch wandb segmentation-models-pytorch torchmetrics optuna torchinfo
pip install torcheval
pip install -e .
```

Download pretrained weights on supercomputer without internet access
```
# Download locally 
python hrmelt/train.py --parallel --no_wandb --cfg_path='runs/unet_smp/sample/config/config.yaml'
# Copy onto supercomputer
scp -r /home/$USER/.cache/huggingface/hub/models--timm--xception71.tf_in1k supercomputer.edu:/path/to/user/.cache/huggingface/hub/ 
# Set offline modes
export WANDB_MODE='offline'
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# Optional: set huggingface cache dir
# export HF_HOME=/path/to/user/.cache/huggingface/hub/
```

#### Debug misc

Evaluation kills processing, running out of memory:
```
The torchmetrics implementation of ssim is know to keep too much in memory. On my laptop that crashes the evaluation. Either update torchmetrics, because I think they fixed this issue. Or, increase memory or pass --exclude_ssim to train.py
```

# Reference
If this analysis is useful for your analysis please consider citing:
```
@misc{lutjens25meltwaterbench,
      title={MeltwaterBench: Deep learning for spatiotemporal downscaling of surface meltwater}, 
      author={Bj\"orn L\"utjens and Patrick Alexander and Raf Antwerpen and Til Widmann and Guido Cervone and Marco Tedesco},
      year={2025},
      journal = {arXiv},
      url = {https://arxiv.org/abs/2512.12142},
}
```
