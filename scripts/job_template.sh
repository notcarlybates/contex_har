#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --output=/scratch/bates.car/jobs/context_har/%j/job_%j.out
#SBATCH --error=/scratch/bates.car/jobs/context_har/%j/job_%j.err
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

cd /home/bates.car/context_har

echo "=== Environment Debug Info ==="
echo "Current directory: $(pwd)"
echo "User: $(whoami)"

module load miniconda3
echo "Miniconda module loaded"

# Clear interference
unset PYTHONPATH
export PYTHONNOUSERSITE=1
echo "Python path cleared"

# Initialize conda
eval "$(conda shell.bash hook)"
echo "Conda initialized"

# Activate environment
conda activate context_har
echo "Activation command executed"

# Check what actually happened
echo "=== Conda Info ==="
conda info --envs
echo ""
echo "Active environment: $CONDA_DEFAULT_ENV"
echo "Which python: $(which python)"
echo "Python version: $(python --version)"
echo "Python path: $(python -c 'import sys; print(sys.executable)')"

# Try to import torch
echo "=== Testing torch import ==="
python -c "import torch; print(f'Torch version: {torch.__version__}')" || echo "FAILED to import torch"

echo "=== Running commands ==="
$1
$2
$3