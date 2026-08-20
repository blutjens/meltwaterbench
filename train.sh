#!/bin/bash

# Throughput script for launching hyperparameter sweep

# Slurm sbatch options. See https://slurm.schedmd.com/archive/slurm-20.02.7/sbatch.html
#SBATCH --output runs/unet_smp/data_v1_4_sensitivity/sweep/task-%a/task.sh.log
#SBATCH --array 9-9 # Create job array. E.g., 1-1 for a single job,
# 1-4 for four jobs, or 5-8 for the next four jobs starting with task ID 5.
#SBATCH --gres=gpu:volta:1 # number of GPU cores
# MIT supercloud std alloc is 8 GPUs and 160 CPU cores. 
# Each cpu core comes with 4GB of memory.
#SBATCH --cpus-per-task 18 # number of workers. 
#SBATCH --time 95:50:00

# Loading the required module
source /etc/profile # eofe: source /etc/profile.d/modules.sh
eval "$(conda shell.bash hook)"
conda deactivate
module load anaconda/2023a-pytorch # eofe: module load anaconda3/2020.11
conda activate hrmelt
export CUBLAS_WORKSPACE_CONFIG=:4096:8 # sets GPU operations like gaussian_kernel to deterministic
export WANDB_MODE='offline'
export TRANSFORMERS_OFFLINE=1 # set hugginface model weights download offline
export HF_DATASETS_OFFLINE=1

# Compute SLURM job array index offset
export SLURM_ARRAY_TASK_OFFSET=$(($SLURM_ARRAY_TASK_MIN - 1))
export TASK_ID_AFTER_OFFSET=$(($SLURM_ARRAY_TASK_ID - $SLURM_ARRAY_TASK_OFFSET))
echo Starting task-$SLURM_ARRAY_TASK_ID
echo Within the current sweep, this is job $TASK_ID_AFTER_OFFSET of $SLURM_ARRAY_TASK_COUNT.

# Run the script
python hrmelt/train.py \
--parallel \
--sweep \
--cfg_path 'runs/unet_smp/data_v1_4_sensitivity/config/config.yaml' \
--task_id $TASK_ID_AFTER_OFFSET \
--num_tasks $SLURM_ARRAY_TASK_COUNT \
--task_id_offset $SLURM_ARRAY_TASK_OFFSET