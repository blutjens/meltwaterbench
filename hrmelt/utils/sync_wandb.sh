#!/bin/bash

# Set the loop interval in seconds (30 minutes = 1800 seconds)
INTERVAL=3600

# Set temporary wandb directory to avoid issues with shared /tmp
export WANDB_DIR=./runs/
echo "Relative root path"
pwd

while true; do
  # Call the wandb sync command. 
  wandb sync --include-offline ./runs/unet_smp/data_v1_4_sensitivity/sweep/task-6/wandb/offline-*
  wandb sync --include-offline ./runs/unet_smp/data_v1_4_sensitivity/sweep/task-7/wandb/offline-*
  wandb sync --include-offline ./runs/unet_smp/data_v1_4_sensitivity/sweep/task-8/wandb/offline-*
  wandb sync --include-offline ./runs/unet_smp/data_v1_4_sensitivity/sweep/task-9/wandb/offline-*
  wandb sync --include-offline ./runs/unet_smp/data_v1_4_hindcast_sensitivity/sweep/task-1/wandb/offline-*

  # Sleep for the interval before looping again
  echo 'sleeping'
  sleep $INTERVAL
done
