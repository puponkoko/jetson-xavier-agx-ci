#!/bin/bash

# Installation script for Llama 4B quantized model on NVIDIA Jetson Xavier AGX
# This script installs all required dependencies and libraries

set -e  # Exit on error

echo "=========================================="
echo "Jetson Xavier AGX - Llama 4B Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo -e "${YELLOW}Warning: This script is designed for NVIDIA Jetson devices${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Get JetPack version
if [ -f /etc/nv_tegra_release ]; then
    JETPACK_VERSION=$(cat /etc/nv_tegra_release | head -c 5)
    echo -e "${GREEN}Detected JetPack version: $JETPACK_VERSION${NC}"
fi

# Update system packages
echo -e "${GREEN}[1/8] Updating system packages...${NC}"
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
echo -e "${GREEN}[2/8] Installing system dependencies...${NC}"
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    python3-pip \
    python3-dev \
    python3-setuptools \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    pkg-config \
    ninja-build \
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    libopenmpi-dev \
    openmpi-bin \
    openmpi-common

# Install CUDA development tools if not present
echo -e "${GREEN}[3/8] Checking CUDA installation...${NC}"
if ! command -v nvcc &> /dev/null; then
    echo -e "${YELLOW}CUDA compiler not found. Installing CUDA toolkit...${NC}"
    sudo apt-get install -y cuda-toolkit-11-4 || sudo apt-get install -y cuda-toolkit-10-2
fi

# Set CUDA paths
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Upgrade pip and install basic Python packages
echo -e "${GREEN}[4/8] Installing Python dependencies...${NC}"
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install --upgrade numpy

# Install PyTorch for Jetson (pre-built wheel)
echo -e "${GREEN}[5/8] Installing PyTorch for Jetson...${NC}"
# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print("".join(map(str, sys.version_info[:2])))')
echo "Python version: $PYTHON_VERSION"

# Install PyTorch - using NVIDIA's pre-built wheels for Jetson
# For JetPack 5.x (Python 3.8+)
if [ "$JETPACK_VERSION" = "R35." ] || [ "$JETPACK_VERSION" = "R36." ]; then
    echo "Installing PyTorch for JetPack 5.x..."
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
elif [ "$JETPACK_VERSION" = "R32." ]; then
    echo "Installing PyTorch for JetPack 4.x..."
    # For JetPack 4.x, use specific version
    wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp38-cp38-linux_aarch64.whl
    python3 -m pip install torch-1.10.0-cp38-cp38-linux_aarch64.whl
    rm torch-1.10.0-cp38-cp38-linux_aarch64.whl
else
    echo -e "${YELLOW}Unknown JetPack version, attempting generic PyTorch installation...${NC}"
    python3 -m pip install torch torchvision torchaudio
fi

# Install Hugging Face libraries
echo -e "${GREEN}[6/8] Installing Hugging Face and ML libraries...${NC}"
python3 -m pip install \
    transformers \
    accelerate \
    sentencepiece \
    protobuf \
    safetensors \
    huggingface-hub \
    tokenizers \
    onnxruntime-gpu \
    psutil

# Install quantization libraries
echo -e "${GREEN}[7/8] Installing quantization libraries...${NC}"
python3 -m pip install \
    bitsandbytes \
    optimum \
    auto-gptq \
    llama-cpp-python

# Build and install llama.cpp from source (for better Jetson optimization)
echo -e "${GREEN}[8/8] Building llama.cpp from source with CUDA support...${NC}"
cd ~
if [ -d "llama.cpp" ]; then
    echo "llama.cpp directory exists, updating..."
    cd llama.cpp
    git pull
else
    echo "Cloning llama.cpp repository..."
    git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp
fi

# Build with CUDA support
make clean
make LLAMA_CUDA=1 -j$(nproc)

# Install Python bindings for llama.cpp
cd ~/llama.cpp
python3 -m pip install -e . --verbose

# Create models directory
mkdir -p ~/models
cd ~/models

echo ""
echo -e "${GREEN}=========================================="
echo "Installation Complete!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Download a quantized Llama model:"
echo "   cd ~/models"
echo "   huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir ."
echo ""
echo "2. Or use llama.cpp to quantize a model:"
echo "   cd ~/llama.cpp"
echo "   ./quantize <model.gguf> <output-q4_0.gguf> q4_0"
echo ""
echo "3. Run inference:"
echo "   cd ~/llama.cpp"
echo "   ./main -m ~/models/llama-2-7b-chat.Q4_K_M.gguf -p 'Your prompt here' -n 128"
echo ""
echo "Or use Python:"
echo "   python3 -c \"from llama_cpp import Llama; llm = Llama(model_path='~/models/model.gguf'); print(llm('Hello', max_tokens=128))\""
echo ""
echo -e "${GREEN}Setup complete!${NC}"

