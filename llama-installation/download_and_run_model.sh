#!/bin/bash

# Script to download and run Llama model on Jetson
# Run this after installation is complete

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Downloading and setting up Llama model...${NC}"
echo ""

# Create models directory
mkdir -p ~/models
cd ~/models

# Check if huggingface tools are installed, install if not
if ! command -v hf &> /dev/null && ! command -v huggingface-cli &> /dev/null; then
    echo -e "${YELLOW}huggingface-cli not found. Installing huggingface-hub...${NC}"
    python3 -m pip install --upgrade huggingface-hub
    # Refresh PATH
    export PATH=$HOME/.local/bin:$PATH
fi

# Download TinyLlama (fastest for testing)
echo -e "${GREEN}Downloading TinyLlama 1.1B model...${NC}"

# Try hf (new command) first, then huggingface-cli, fallback to Python
if command -v hf &> /dev/null; then
    hf download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --local-dir .
elif command -v huggingface-cli &> /dev/null; then
    huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --local-dir .
else
    echo -e "${YELLOW}Using Python to download model...${NC}"
    python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF', filename='tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf', local_dir='.')"
fi

echo ""
echo -e "${GREEN}Model downloaded!${NC}"
echo ""
echo -e "${YELLOW}Running test inference...${NC}"
echo ""

# Check if llama.cpp exists, otherwise use Python
if [ -f ~/llama.cpp/main ]; then
    # Use llama.cpp (C++ - fastest)
    cd ~/llama.cpp
    ./main -m ~/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -p "Hello, how are you?" -n 50 -t $(nproc)
else
    # Use Python if llama.cpp not built
    echo -e "${YELLOW}llama.cpp not found, using Python instead...${NC}"
    python3 -c "from llama_cpp import Llama; import os; llm = Llama(model_path=os.path.expanduser('~/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf'), n_gpu_layers=35); print(llm('Hello, how are you?', max_tokens=50)['choices'][0]['text'])"
fi

echo ""
echo -e "${GREEN}Done!${NC}"
echo ""
echo "To run again:"
if [ -f ~/llama.cpp/main ]; then
    echo "  cd ~/llama.cpp"
    echo "  ./main -m ~/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -p 'Your prompt' -n 100 -t \$(nproc)"
else
    echo "  python3 -c \"from llama_cpp import Llama; import os; llm = Llama(model_path=os.path.expanduser('~/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf'), n_gpu_layers=35); print(llm('Your prompt', max_tokens=100)['choices'][0]['text'])\""
    echo ""
    echo "  Or create an interactive chat script:"
    echo "  python3 ~/example_llama_inference.py"
fi

