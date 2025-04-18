#!/bin/bash
#SBATCH --job-name=llama_download
#SBATCH --output=llama_download.out
#SBATCH --error=llama_download.err
#SBATCH --time=01:00:00
#SBATCH --partition=class
#SBATCH --account=class_cse476spring2025
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

module load mamba/latest
source activate bentune

echo "Using environment: $CONDA_DEFAULT_ENV"
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

echo "Starting vanilla_loader.py..."
python3 model/vanilla_download/vanilla_loader.py
echo "Model download complete."
