#!/bin/bash
#SBATCH --job-name=llama_download
#SBATCH --output=output_logs/vanilla_download.out
#SBATCH --error=output_logs/vanilla_download.err
#SBATCH --time=01:00:00
#SBATCH -p general
#SBATCH -A class_cse476spring2025
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

module load mamba/latest
source activate bentune

echo "Using environment: $CONDA_DEFAULT_ENV"
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

SAVE_DIR=downloaded_models/vanilla-llama-3.2-3b-bf16
mkdir -p $SAVE_DIR

echo "Starting vanilla_loader.py..."
python3 model/vanilla_download/bf16/vanilla_loader.py --save_dir $SAVE_DIR
echo "Model download complete."