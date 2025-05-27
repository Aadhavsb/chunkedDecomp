#!/bin/bash

# Load required modules (adjust based on your cluster)
module load python/3.9
module load cuda/11.8
module load gcc/9.3.0

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support (adjust CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt

# Install project in development mode
pip install -e .

echo "Environment setup complete!"
