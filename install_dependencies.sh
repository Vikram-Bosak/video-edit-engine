#!/bin/bash
set -euo pipefail

echo "========================================="
echo "  AI Video Edit Engine - Dependency Setup"
echo "========================================="

echo "[1/4] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    ffmpeg \
    imagemagick \
    libass-dev \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    > /dev/null 2>&1

echo "[2/4] Upgrading pip..."
python -m pip install --upgrade pip --quiet

echo "[3/4] Installing Python dependencies..."
pip install -r requirements.txt --quiet

echo "[4/4] Verifying installations..."
echo "  FFmpeg: $(ffmpeg -version 2>&1 | head -1)"
echo "  Python: $(python --version)"
python -c "import cv2; print(f'  OpenCV: {cv2.__version__}')"
python -c "import numpy; print(f'  NumPy: {numpy.__version__}')"
python -c "from PIL import Image; print('  Pillow: OK')"
python -c "import yaml; print('  PyYAML: OK')"

echo ""
echo "========================================="
echo "  All dependencies installed successfully"
echo "========================================="
