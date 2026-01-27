#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --output=/scratch/bates.car/jobs/metrics_%j/metrics_job_%j.out
#SBATCH --error=/scratch/bates.car/jobs/metrics_%j/metrics_job_%j.err
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

cd /home/bates.car/context_har

module load miniconda3

# Clear interference
unset PYTHONPATH
export PYTHONNOUSERSITE=1

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate existing environment
conda activate context_har

# Run your command
python main.py --config ./configs/main_experiments/deepconvcontext/hangtime_loso_bilstm.yaml --seed 1