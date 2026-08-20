"""
source: https://github.com/milesial/Pytorch-UNet/blob/master/train.py
edited by Björn Lütjens
"""
import copy

import yaml
import argparse
import logging
import os
import random
import time
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from osgeo import gdal  # rasterio in dataloader uses
# gdal. Need to import gdal to suppress warning msg.
from pathlib import Path
from pprint import pprint
from torch import optim
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm
from typing import Callable, Dict, List, Any
from functools import partial

import wandb

from hrmelt.eval.benchmark import benchmark_metrics
from hrmelt.eval.benchmark import log_benchmark_metrics_to_wandb
from hrmelt.eval.metrics import MaskedR2, MaskedSSIM, CountValidPx
from hrmelt.evaluate import evaluate
from hrmelt.predict import predict 
from hrmelt.predict import HRMeltDatasetConvolution
from hrmelt.dataset import HRMeltDataset
from hrmelt.dataset import batch_preprocess_fn
from hrmelt.utils.utils import lookup_torch_dtype
from hrmelt.utils.utils import MaskedLoss
from hrmelt.utils.utils import set_num_workers
from hrmelt.utils.utils import init_sweep_config
from hrmelt.utils.utils import set_all_seeds
from hrmelt.utils.utils import _worker_init_fn

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        save_checkpoint: bool = True,
        amp: bool = False,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        num_workers: int = None,
        no_wandb: bool = False,
        parallel: bool = False,
        sweep: bool = False,
        cfg: dict = None,
):
    # Create dataset and train, val, test partition.
    dtype = lookup_torch_dtype(cfg['dtype'])
    if cfg['split_cfg'] == 'csv':
        # Initialize dataset from .csv file with filenames
        train_set = HRMeltDataset(cfg=cfg, split='train', verbose=False)
        val_set = HRMeltDataset(cfg=cfg, split='val', verbose=False)
    else:
        # Initialize dataset from providing a data root folder
        dataset = HRMeltDataset(cfg=cfg, verbose=False)
        train_set, val_set, _ = dataset.create_data_splits(
            cfg['split_cfg'],
            val_percent=cfg['val_percent'],
            test_percent=cfg['test_percent'],
            seed=cfg['seed'],
            create_csv_w_split_cfg=cfg['create_csv_w_split_cfg'],
            path_train_split_csv=cfg['path_train_split_csv'],
            path_val_split_csv=cfg['path_val_split_csv'],
            path_test_split_csv=cfg['path_test_split_csv'],
            path_all_split_csv=cfg['path_all_split_csv'] if 'path_all_split_csv' in cfg else None)

    # Create data loader
    num_workers = set_num_workers(num_workers, parallel=parallel)

    # Enable gdal error messages (otherwise each CPU raises a warning message at the start of each epoch)
    gdal.UseExceptions()

    # init_fn needs to be global func so it can be Pickeled
    init_fn_with_cfg = partial(_worker_init_fn, seed=cfg['seed'])

    batch_preprocess_partial_fn = partial(batch_preprocess_fn, cfg=cfg, device=device)

    train_drop_last = False
    if 'batch_norm' in cfg.keys():
        if cfg['batch_norm'] == True or batch_size == 1:
            # drop last batch to avoid batch_size = 1 if batch norm is used
            train_drop_last = True 
    loader_args = dict(batch_size=batch_size,
                       num_workers=num_workers,
                       pin_memory=True,
                       worker_init_fn=init_fn_with_cfg,
                       collate_fn=batch_preprocess_partial_fn)
    persistent_workers_train = cfg['persistent_workers'] if 'persistent_workers' in cfg.keys() else False
    train_loader = DataLoader(train_set, shuffle=True, drop_last=train_drop_last, 
                              persistent_workers=persistent_workers_train,
                              **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)
    
    if cfg['periodical_evaluation']:
        logging.info("Periodical evaluation active.")

        loader_args['batch_size'] = cfg['prediction_batch_size']
        predict_set = HRMeltDatasetConvolution(cfg=cfg,
                                               split='eval',
                                               verbose=False,
                                               stride=cfg['prediction_stride'])
        predict_loader = DataLoader(predict_set, shuffle=False, drop_last=False, **loader_args)
    
    # (Initialize logging)
    if not no_wandb:
        wandb_run = wandb.init(entity=cfg['wandb_entity'],
                               project=cfg['wandb_project_name'],
                               resume='allow', anonymous='must',
                               dir=cfg['path_wandb'])

        wandb_cfg_log = dict(epochs=epochs, 
                 batch_size=batch_size, 
                 num_workers=num_workers,
                 img_size=cfg['img_size'][0],
                 optimizer=cfg['optimizer'],
                 seed=cfg['seed'],
                 learning_rate=learning_rate,
                 weight_decay = cfg['weight_decay'],
                 loss_function = cfg['loss_function'],
                 train_len = train_set.__len__(),
                 val_len = val_set.__len__(),
                 val_percent=cfg['val_percent'], 
                 lr_patience=cfg['lr_patience'], # for step lr_scheduler
                 lr_scheduler=cfg['lr_scheduler'],
                 T_0=cfg['T_0'], # for cosine lr scheduler
                 T_mult=cfg['T_mult'], # for cosine lr scheduler
                 eta_min=cfg['eta_min'], # for cosine lr scheduler
                 pmw_GaussianBlur_kernel_size=cfg['pmw_GaussianBlur_kernel_size'],
                 pmw_GaussianBlur_sigma=cfg['pmw_GaussianBlur_sigma'],
                 mar_wa1_GaussianBlur_kernel_size=cfg['mar_wa1_GaussianBlur_kernel_size'],
                 mar_wa1_GaussianBlur_sigma=cfg['mar_wa1_GaussianBlur_sigma'],
                 save_checkpoint=save_checkpoint,
                 path_checkpoints=cfg['path_checkpoints'], # add checkpoint paths to find checkpoints after large sweep
                 in_keys=cfg['in_keys'],
                 amp=amp)
        for hyperparam in cfg['model_args']:
            if isinstance(cfg[hyperparam], list):
                # If hyperparameter is a list enter each entry separately
                for i, entry in enumerate(cfg[hyperparam]):
                    wandb_cfg_log[f'{hyperparam}_{i}'] = entry
            else:
                wandb_cfg_log[hyperparam] = cfg[hyperparam]
        wandb_run.config.update(wandb_cfg_log)
        # Log full config file to wandb -> artifacts -> config-file -> Files
        artifact = wandb.Artifact(name="config-file", type="config")
        artifact.add_file(local_path=cfg['path_cfg'], name="config.yaml")
        wandb_run.log_artifact(artifact)  # Logs the cfg to "config.yaml:v0"
        # wandb.save(args.cfg_path) # I believe this only saves the cfg path.
    else:
        wandb_run = None

    logging_info_str = f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Image size:      {cfg['img_size']}        
        Optimizer:       {cfg['optimizer']}
        Learning rate:   {learning_rate}
        LR Patience:     {cfg['lr_patience']}
        Weight decay:    {cfg['weight_decay']}
        Loss function:   {cfg['loss_function']}
        In channels:     {cfg['in_channels']}
        Activation:      {cfg['activation']}
        Out Activation:  {cfg['out_activation']}
        Training size:   {train_set.__len__()}
        Validation size: {val_set.__len__()}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
        Num. Workers:    {num_workers}
        In keys:         {cfg['in_keys']}
        In keys static:  {cfg['in_keys_static']}
        In keys aux:     {cfg['in_keys_aux']}
    '''
    for hyperparam in cfg['model_args']:
        logging_info_str += f'{hyperparam}  \t: {cfg[hyperparam]}\n'
    logging.info(logging_info_str)

    # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    if cfg['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                            lr=learning_rate, weight_decay=cfg['weight_decay'], 
                            foreach=True)
    elif cfg['optimizer'] == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(),
                            lr=learning_rate, weight_decay=cfg['weight_decay'], 
                            momentum=momentum, foreach=True)
    else:
        optimizer = optim.Adam(model.parameters(), 
                            lr=learning_rate, betas=(0.9, 0.999), 
                            weight_decay=cfg['weight_decay'], foreach=True)

    if cfg['lr_scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=cfg['T_0'],
            T_mult=cfg['T_mult'],
            eta_min=cfg['eta_min'],
        )
    else: # cfg['lr_scheduler'] == 'ReduceLROnPlateau'
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                     patience=cfg['lr_patience'])

    if amp and device_type != 'cuda':
        logging.warning(
            f"AMP requested but device is '{device_type}'; GradScaler is only "
            "supported on CUDA, so it will be disabled."
        )
        amp = False
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    criterion = MaskedLoss(cfg['loss_function'])
    # criterion = nn.CrossEntropyLoss() if model.out_channels > 1 else nn.BCEWithLogitsLoss()
 
    if cfg['periodical_evaluation']:
        metrics_fn = {
            'MaskedL1': MaskedLoss('l1', reduction='none'),
            'MaskedMSE': MaskedLoss('mse', reduction='none'),
            'MaskedR2': MaskedR2(reduction='none'),
            'ValidPx': CountValidPx(reduction='none')
        }
    if cfg['periodical_evaluation_include_ssim']:
        # SSIM computation requires a lot of memory
        metrics_fn['MaskedSSIM'] = MaskedSSIM(data_range = (0.,1.), sigma=10., device=device)        
    if args.exclude_ssim and 'MaskedSSIM' in metrics_fn:
        del metrics_fn['MaskedSSIM']

    # Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_set),
                  desc=f'Epoch {epoch}/{epochs}', unit='img',
                  #disable=(sweep==False) # disable tqdm if printing to log instead of console
                  ) as pbar:
            for i, batch in enumerate(train_loader):
                batch_start = time.time()
                inputs, targets, targets_mask, meta = batch
                assert inputs.shape[1] == model.get_in_channels(), \
                    f'Network has been defined with {model.get_in_channels()} input channels, ' \
                    f'but loaded images have {inputs.shape[1]} channels. Please check that ' \
                    'the inputs are loaded correctly.'

                if cfg['model_key'] == 'linear_dem':
                    if inputs.shape[0] == 1:
                        print('DEBUGGING linear model; remove this code later on')
                        months = inputs[0,-1,0,0].cpu().detach().numpy().astype(int)
                        print('months: ', months)
                        print('weights and biases:')
                        weights = model.linear_layers[months-1]._parameters['weight'].cpu().detach().numpy().squeeze()
                        bias = model.linear_layers[months-1]._parameters['bias'].cpu().detach().numpy().squeeze()
                        print('w: ', weights)
                        print('b: ', bias)
                        y_decision_boundary = 0.5
                        if 'time_sin' in cfg['in_keys_aux']:
                            sin_time = inputs[0,1].unique().cpu().numpy()
                            cos_time = inputs[0,2].unique().cpu().numpy()
                            dem_dec_bnd_unnormalized = (y_decision_boundary - weights[1] * sin_time - weights[2] * cos_time - bias) / weights[0]
                        else:
                            dem_dec_bnd_unnormalized = (y_decision_boundary - bias) / weights
                        dem_dec_bnd = (dem_dec_bnd_unnormalized * cfg['std_dem']) + cfg['mean_dem']
                        print(f'DEM decision boundary for month {months}: {dem_dec_bnd:.2f}m.')
                        if cfg['out_activation'] == 'tanhshelf':
                            print(f'tanhshelf sharpness {model.out_act.sharpness.cpu().detach().numpy().squeeze():.3f}')
                        elif cfg['out_activation'] == 'sigmoidcutoff':
                            print(f'sigmoidcutoff sharpness {model.out_act.sharpness.cpu().detach().numpy().squeeze():.3f}')
                # todo: check if these need requires_grad = true
                inputs = inputs.to(device=device, dtype=dtype, memory_format=torch.channels_last)
                targets = targets.to(device=device, dtype=dtype)
                targets_mask = targets_mask.to(device=device, dtype=dtype)
                inference_start = time.time()

                if device.type == 'mps':
                    pred = model(inputs)
                    loss = criterion(pred, targets, targets_mask)

                else:
                    with torch.autocast(device.type, enabled=amp):
                        pred = model(inputs)
                        loss = criterion(pred, targets, targets_mask)

                inference_end = time.time()
                
                grad_start = time.time()
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                grad_end = time.time()
                if cfg['lr_scheduler'] == 'CosineAnnealingWarmRestarts':
                    # Set step to a real value between 0 and cfg['epochs']
                    scheduler.step((epoch-1) + (i / len(train_loader)))
                
                pbar.update(inputs.shape[0])
                epoch_loss += loss.item()
                batch_end = time.time()
                if not no_wandb:
                    # Commit is True, which will increase the global wandb step for every batch.
                    wandb_run.log({
                        'train loss': loss.item(),
                        'learning rate': optimizer.param_groups[0]['lr'],
                        'batch_time': batch_end - batch_start,
                        'inference_time': inference_end - inference_start,
                        'gradient_time': grad_end - grad_start
                    }, commit=True)
                pbar.set_postfix(**{'avg loss/img': epoch_loss / float(i+1)})

        # Evaluation round
        # division_step = (len(train_loader) // (5 * batch_size))
        # global_step = (len(train_loader) * epoch-1)
        if 1: # division_step > 0:
            if 1:  # global_step % division_step == 0:
                val_score = evaluate(model, val_loader, criterion, 
                                     device, amp, dtype, cfg=cfg,
                                     wandb_run=wandb_run)
                if cfg['lr_scheduler'] == 'ReduceLROnPlateau':
                    scheduler.step(val_score)

        # Save model
        epochs_between_checkpoint_save = cfg['epochs_between_checkpoint_save'] if 'epochs_between_checkpoint_save' in cfg.keys() else 10
        if ((save_checkpoint and 
            epoch >= cfg['starting_epoch'] and
            (epoch-cfg['starting_epoch']) % epochs_between_checkpoint_save == 0) or
            (save_checkpoint and 
             epoch == epochs)):
            Path(cfg['path_checkpoints']).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            checkpoint_filename = f'checkpoint_epoch{epoch}.pth'
            if 'add_wandb_id_to_ckpt_filename' in cfg.keys():
                if 'add_wandb_id_to_ckpt_filename' == True:
                    run_id = f'{wandb.run.id}_' if not no_wandb else 'no_wandb_'
                    checkpoint_filename = f'{run_id}' + checkpoint_filename
            torch.save(state_dict, str(Path(cfg['path_checkpoints']) / checkpoint_filename))
            # torch.load(str(Path(cfg['path_checkpoints']) / checkpoint_filename))
            logging.info(f'Checkpoint {epoch} saved!')

        # Evaluate the model on predictions over the full-scale tif using multiple metrics and log it to wandb
        if ((cfg['periodical_evaluation'] and 
            (epoch-cfg['starting_epoch']) % cfg['epochs_between_periodical_evaluation'] == 0 and
                epoch >= cfg['starting_epoch']) or 
            (cfg['periodical_evaluation'] and
             epoch == epochs)):
            logging.info("Starting extensive evaluation.")

            predict(
                model=model,
                dataloader=predict_loader,
                device=device,
                cfg=cfg,
                compress=True,
                specific_pred_path=cfg['path_eval'],
                verbose=(epoch==1)
            )

            # creating copy because benchmark metrics mutates cfg
            cfg_copy = copy.deepcopy(cfg)
            report = benchmark_metrics(
                metrics_fn,
                [cfg['model_key']],
                device,
                cfg_copy,
                split='eval',
                return_filenames=True,
                verbose=(epoch==1))

            if not no_wandb:
                log_benchmark_metrics_to_wandb(metrics=report, wandb_run=wandb_run, metrics_fn_keys=metrics_fn.keys())

            # Log full-scale images to wandb during selected evaluations
            period_wandb_full_scale_im = cfg['period_wandb_full_scale_im'] if 'period_wandb_full_scale_im' in cfg.keys() else 1
            if (not no_wandb and 
                ((epoch-cfg['starting_epoch']) // cfg['epochs_between_periodical_evaluation']) % period_wandb_full_scale_im == 0):
                # get paths of prediction pngs
                predictions = []
                for filename in sorted(os.listdir(cfg['path_eval'])):
                    if filename.endswith('.png'):
                        file_path = os.path.join(cfg['path_eval'], filename)
                        predictions.append(file_path)

                # Log compressed full-scale images to wandb
                wandb_run.log(
                    {'evaluation_preds': [wandb.Image(file, caption=Path(file).stem)
                                            for file in predictions]}, commit=False)

            del report
            logging.info("Done")

        # Log epoch at the end of the epoch to make sure evalution is logged into the correct epoch
        if not no_wandb:
            wandb_run.log({'epoch': epoch}, commit=False)

    print("Finished train.py")

def get_args():
    parser = argparse.ArgumentParser(description='Train the model on images and target masks')
    parser.add_argument('--load', '-f', type=str, default=None, help='Load model from a .pth file')
    parser.add_argument('--cfg_path', type=str, default='runs/unet/default/config/config.yaml',
                        help='Path to config yaml')
    # For a different model pass, e.g., 'runs/deeplabv3/default/config/config.yaml'
    parser.add_argument('--parallel', action='store_true', default=False, help='Enable parallel training')
    parser.add_argument('--no_wandb', action='store_true', default=False, help='Disable wandb logs')
    parser.add_argument('--verbose', type=bool, default=False, help='Set true to print verbose logs')
    parser.add_argument('--exclude_ssim', action='store_true', help='For debugging w/o ssim as it takes excessive memory')
    parser.add_argument('--task_id', type=int, default=1, help='SLURM task id, when script is called in job array')
    parser.add_argument('--task_id_offset', type=int, default=0, help='Start filepaths for tasks at this ID. '\
                        'Used to add a second hyperparam sweep into directory with existing sweep')
    parser.add_argument('--num_tasks', type=int, default=1,
                        help='Total number of SLURM tasks when script is called in job array')
    parser.add_argument('--sweep', action='store_true', default=False,
                        help='If true, indicates that program is running a hyperparameter sweep')
    return parser.parse_args()

if __name__ == '__main__':
    # Get command line arguments
    args = get_args()

    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Import cfg
    cfg = yaml.safe_load(open(args.cfg_path, 'r'))
    cfg['path_cfg'] = args.cfg_path

    # Init device
    if cfg['device'] == 'gpu':
        if torch.cuda.is_available():
            device_type = 'cuda'
        elif torch.backends.mps.is_available():
            device_type = 'mps'
        else:
            device_type = 'cpu'
    else:
        device_type = 'cpu'
    device = torch.device(torch.device(device_type))
    logging.info(f'Using device {device}')

    # Necessary to send variables to GPU in dataloader collate_fn 
    if cfg['use_batch_blur']:
        torch.multiprocessing.set_start_method('spawn')
    
    # Set seeds
    set_all_seeds(cfg['seed'], device=device.type, 
                  use_deterministic_algorithms=cfg['use_deterministic_algorithms'],
                  warn_only=cfg['warn_only'])

    if args.verbose:
        print('Default model configuration:')
        pprint(cfg)

    # Initialize hyperparameter sweep
    if args.sweep:
        cfg = init_sweep_config(cfg, cfg['path_sweep_cfg'], args.task_id, args.num_tasks, args.task_id_offset)
    
    cfg['in_channels'] = len(cfg['in_keys']) + len(cfg['in_keys_static']) + len(cfg['in_keys_aux'])

    if cfg['model_key']=='unet':
        from hrmelt.models.unet.unet_model import UNet
        cfg['model_args'] = {
            'in_channels': cfg['in_channels'],
            'activation': cfg['activation'],
            'out_activation': cfg['out_activation'],
            'out_channels': cfg['out_channels'],
            'bilinear': cfg['bilinear'],
            'num_extra_convs': cfg['num_extra_convs'],
        }
        model = UNet(**cfg['model_args'])
    elif cfg['model_key']=='unet_smp':
        from hrmelt.models.unet_smp.model import UNet_smp
        cfg['model_args'] = {
            'in_channels': cfg['in_channels'],
            'out_activation': cfg['out_activation'],
            'encoder_name': cfg['encoder_name'],
            'encoder_depth': cfg['encoder_depth'],
            'encoder_weights': cfg['encoder_weights'],
            'decoder_use_batchnorm': cfg['decoder_use_batchnorm'],
            'out_channels': cfg['out_channels'],
        }
        model = UNet_smp(**cfg['model_args'])
    elif cfg['model_key']=='deeplabv3':
        from hrmelt.models.deeplabv3.deeplabv3_model import DeepLabV3Plus
        # rename some model hyperparameters
        cfg['classes'] = cfg['out_channels']
        # List all model hyperparameters
        cfg['model_args'] = {
            'in_channels': cfg['in_channels'],  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            'activation': cfg['activation'],  # activation for decoder layers
            'out_activation': cfg['out_activation'],  # output activation function
            'encoder_name': cfg['encoder_name'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            'encoder_depth': cfg['encoder_depth'],  # input size has to be at least dividable by 2 encoder_depth times
            'encoder_weights': cfg['encoder_weights'],  # use `imagenet` pre-trained weights for encoder initialization
            'encoder_output_stride': cfg['encoder_output_stride'],
            'decoder_channels': cfg['decoder_channels'],  # if encoder depth or image size is changed this has to be updated
            'decoder_atrous_rates': cfg['decoder_atrous_rates'],
            'classes': cfg['classes'],  # model output channels (number of classes in your dataset)
            'batch_norm': cfg['batch_norm'],  # batch normalization active or not
            'upsampling': cfg['upsampling']
        }
        model = DeepLabV3Plus(**cfg['model_args'])
    elif cfg['model_key'] == 'linear_dem':
        from hrmelt.models.linear.linear_model import Linear
        cfg['model_args'] = {
            'in_channels': cfg['in_channels'],
            'out_activation': cfg['out_activation'],
            'out_channels': cfg['out_channels'],
            'model_key': cfg['model_key'],
            'n_models': cfg['n_models'],
            'time_channel_idx': cfg['time_channel_idx']
        }
        model = Linear(**cfg['model_args'])

    model = model.to(memory_format=torch.channels_last)

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)

    train_args = {
        'model' : model,
        'epochs' : cfg['epochs'],
        'batch_size' : cfg['batch_size'],
        'learning_rate' : cfg['learning_rate'],
        'device' : device,
        'amp' : cfg['amp'],
        'num_workers' : cfg['num_workers'],
        'parallel' : args.parallel,
        'no_wandb' : args.no_wandb,
        'sweep' : args.sweep,
        'cfg' : cfg
    }
    try:
        train_model(**train_args)
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(**train_args)
