# Jetson Xavier AGX - Llama 4B Setup Guide

This guide will help you install all required dependencies to run a quantized Llama 4B model on your NVIDIA Jetson Xavier AGX.

## Quick Start

### Option 1: Automated Installation (Recommended)

1. **Transfer the installation script to your Jetson device:**
   ```bash
   scp install_jetson_llama.sh user@jetson-ip:~/
   ```

2. **SSH into your Jetson device:**
   ```bash
   ssh user@jetson-ip
   ```

3. **Run the installation script:**
   ```bash
   chmod +x ~/install_jetson_llama.sh
   ~/install_jetson_llama.sh
   ```

   The script will:
   - Update system packages
   - Install all system dependencies
   - Install PyTorch optimized for Jetson
   - Install Hugging Face libraries
   - Build llama.cpp with CUDA support
   - Set up the environment

### Option 2: Manual Installation

If you prefer to install step by step, follow the sections below.

## System Requirements

- NVIDIA Jetson Xavier AGX
- JetPack 4.6+ or JetPack 5.x
- Ubuntu 18.04/20.04 (JetPack 4.x) or Ubuntu 22.04 (JetPack 5.x)
- At least 16GB of storage space
- Internet connection

## Step-by-Step Installation

### 1. Update System

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### 2. Install System Dependencies

```bash
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
```

### 3. Install PyTorch for Jetson

**For JetPack 5.x (Python 3.8+):**
```bash
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For JetPack 4.x:**
```bash
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp38-cp38-linux_aarch64.whl
python3 -m pip install torch-1.10.0-cp38-cp38-linux_aarch64.whl
rm torch-1.10.0-cp38-cp38-linux_aarch64.whl
```

### 4. Install Python ML Libraries

```bash
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements_jetson.txt
```

Or install individually:
```bash
python3 -m pip install \
    transformers \
    accelerate \
    sentencepiece \
    protobuf \
    safetensors \
    huggingface-hub \
    tokenizers \
    bitsandbytes \
    optimum \
    auto-gptq \
    llama-cpp-python \
    onnxruntime-gpu \
    psutil
```

### 5. Build llama.cpp with CUDA Support

```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make clean
make LLAMA_CUDA=1 -j$(nproc)
python3 -m pip install -e . --verbose
```

## Downloading and Using Models

### Option 1: Download Pre-quantized Model

```bash
mkdir -p ~/models
cd ~/models

# Using huggingface-cli
huggingface-cli login  # Login first if needed
huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir .

# Or download a 4B model
huggingface-cli download QuantFactory/Llama-3.1-Minitron-4B-Width-Base-GGUF --local-dir .
```

### Option 2: Quantize Your Own Model

```bash
cd ~/llama.cpp
./quantize <input-model.gguf> <output-q4_0.gguf> q4_0
```

Available quantization levels:
- `q4_0` - 4-bit quantization (fastest, lower quality)
- `q4_1` - 4-bit quantization with improved quality
- `q5_0` - 5-bit quantization
- `q5_1` - 5-bit quantization with improved quality
- `q8_0` - 8-bit quantization (better quality, larger size)

## Running Inference

### Using llama.cpp (C++ - Fastest)

```bash
cd ~/llama.cpp
./main -m ~/models/your-model.gguf -p "Your prompt here" -n 128 -t 8
```

Parameters:
- `-m`: Model path
- `-p`: Prompt
- `-n`: Number of tokens to generate
- `-t`: Number of threads

### Using Python with llama-cpp-python

```python
from llama_cpp import Llama

# Load model
llm = Llama(
    model_path="~/models/your-model.gguf",
    n_ctx=2048,  # Context window
    n_threads=8,  # Number of CPU threads
    n_gpu_layers=35  # Number of layers to offload to GPU
)

# Generate text
response = llm("Your prompt here", max_tokens=128)
print(response['choices'][0]['text'])
```

### Using Transformers (Hugging Face)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "meta-llama/Llama-2-7b-chat-hf"  # Or your model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True  # 4-bit quantization
)

prompt = "Your prompt here"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Performance Optimization Tips

1. **Use GPU offloading:** Set `n_gpu_layers` in llama-cpp-python to offload layers to GPU
2. **Adjust thread count:** Use `nproc` to get CPU count: `./main -t $(nproc)`
3. **Use appropriate quantization:** Q4_0 for speed, Q8_0 for quality
4. **Set power mode:** Use `sudo nvpmodel -m 0` for maximum performance
5. **Enable jetson_clocks:** `sudo jetson_clocks` for maximum clock speeds

## Troubleshooting

### CUDA Out of Memory
- Reduce `n_gpu_layers` in llama-cpp-python
- Use smaller context window (`n_ctx`)
- Use lower quantization (Q4_0 instead of Q8_0)

### Slow Inference
- Ensure CUDA is being used: Check with `nvidia-smi`
- Increase `n_gpu_layers` to offload more to GPU
- Use `sudo jetson_clocks` to maximize clock speeds
- Check power mode: `sudo nvpmodel -q`

### Build Errors
- Ensure all system dependencies are installed
- Check CUDA version: `nvcc --version`
- Try building with fewer threads: `make LLAMA_CUDA=1 -j4`

## Recommended Models for Jetson Xavier AGX

- **Llama-2-7B-Chat-GGUF** (Q4_K_M): Good balance of quality and speed
- **Llama-3.1-Minitron-4B**: Smaller, faster, good for Jetson
- **Mistral-7B-Instruct-GGUF** (Q4_K_M): Alternative option

## Additional Resources

- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [Hugging Face Models](https://huggingface.co/models)
- [Jetson Generative AI Playground](https://tokk-nv.github.io/jetson-generative-ai-playground/)

## Verification

Test your installation:

```bash
# Test PyTorch
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Test llama.cpp
cd ~/llama.cpp
./main -h

# Test Python bindings
python3 -c "from llama_cpp import Llama; print('llama-cpp-python installed successfully')"
```

