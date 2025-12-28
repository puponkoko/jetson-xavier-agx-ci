#!/bin/bash

# Quick-start script for running Llama 4B on Jetson after installation
# Run this on your Jetson device

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=========================================="
echo "Jetson Llama 4B - Next Steps"
echo "==========================================${NC}"
echo ""

# Step 1: Verify Installation
echo -e "${GREEN}[Step 1] Verifying installation...${NC}"

echo "Checking PyTorch..."
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__} installed'); print(f'✓ CUDA available: {torch.cuda.is_available()}')" || echo "✗ PyTorch not found"

echo "Checking llama.cpp..."
if [ -f ~/llama.cpp/main ]; then
    echo "✓ llama.cpp built successfully"
    ~/llama.cpp/main -h | head -5
else
    echo "✗ llama.cpp not found. Run: cd ~/llama.cpp && make LLAMA_CUDA=1"
fi

echo "Checking Python libraries..."
python3 -c "from llama_cpp import Llama; print('✓ llama-cpp-python installed')" 2>/dev/null || echo "✗ llama-cpp-python not installed"
python3 -c "import transformers; print('✓ transformers installed')" 2>/dev/null || echo "✗ transformers not installed"

echo ""

# Step 2: Create models directory
echo -e "${GREEN}[Step 2] Setting up models directory...${NC}"
mkdir -p ~/models
cd ~/models
echo "✓ Models directory ready at ~/models"
echo ""

# Step 3: Download model options
echo -e "${GREEN}[Step 3] Choose a model to download:${NC}"
echo ""
echo "Option A: Llama 3.1 Minitron 4B (Recommended for Jetson - smaller, faster)"
echo "  Command:"
echo "  cd ~/models"
echo "  hf download QuantFactory/Llama-3.1-Minitron-4B-Width-Base-GGUF --local-dir ."
echo "  # Or: huggingface-cli download ... (if hf not available)"
echo ""
echo "Option B: Llama 2 7B Chat (Larger, better quality)"
echo "  Command:"
echo "  cd ~/models"
echo "  hf download TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir ."
echo ""
echo "Option C: TinyLlama 1.1B (Fastest, smallest) - RECOMMENDED FOR FIRST TEST"
echo "  Command:"
echo "  cd ~/models"
echo "  hf download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --local-dir ."
echo ""

read -p "Would you like to download a model now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Which model? (A/B/C): "
    read -n 1 model_choice
    echo
    
    # Check if hf or huggingface-cli is available
    if command -v hf &> /dev/null; then
        DL_CMD="hf download"
    elif command -v huggingface-cli &> /dev/null; then
        DL_CMD="huggingface-cli download"
    else
        echo "Installing huggingface-hub..."
        python3 -m pip install --upgrade huggingface-hub
        export PATH=$HOME/.local/bin:$PATH
        if command -v hf &> /dev/null; then
            DL_CMD="hf download"
        else
            DL_CMD="huggingface-cli download"
        fi
    fi
    
    case $model_choice in
        [Aa]*)
            echo "Downloading Llama 3.1 Minitron 4B..."
            $DL_CMD QuantFactory/Llama-3.1-Minitron-4B-Width-Base-GGUF --local-dir .
            MODEL_FILE=$(find . -name "*.gguf" | head -1)
            ;;
        [Bb]*)
            echo "Downloading Llama 2 7B Chat..."
            $DL_CMD TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir .
            MODEL_FILE="llama-2-7b-chat.Q4_K_M.gguf"
            ;;
        [Cc]*)
            echo "Downloading TinyLlama 1.1B..."
            $DL_CMD TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --local-dir .
            MODEL_FILE="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
            ;;
        *)
            echo "Invalid choice. Skipping download."
            MODEL_FILE=""
            ;;
    esac
    
    if [ -n "$MODEL_FILE" ] && [ -f "$MODEL_FILE" ]; then
        echo -e "${GREEN}✓ Model downloaded: $MODEL_FILE${NC}"
        echo ""
        echo -e "${YELLOW}Ready to run inference!${NC}"
        echo ""
        if [ -f ~/llama.cpp/main ]; then
            echo "Quick test command (using llama.cpp):"
            echo "  cd ~/llama.cpp"
            echo "  ./main -m ~/models/$MODEL_FILE -p 'Hello, how are you?' -n 50 -t \$(nproc)"
        else
            echo "Quick test command (using Python):"
            echo "  python3 -c \"from llama_cpp import Llama; import os; llm = Llama(model_path=os.path.expanduser('~/models/$MODEL_FILE'), n_gpu_layers=35); print(llm('Hello, how are you?', max_tokens=50)['choices'][0]['text'])\""
        fi
    fi
fi

echo ""
echo -e "${BLUE}=========================================="
echo "Quick Reference Commands"
echo "==========================================${NC}"
echo ""
echo "1. Run inference with llama.cpp (C++ - fastest, if built):"
echo "   cd ~/llama.cpp"
echo "   ./main -m ~/models/your-model.gguf -p 'Your prompt' -n 128 -t \$(nproc)"
echo ""
echo "2. Run inference with Python (works if llama-cpp-python installed):"
echo "   python3 -c \"from llama_cpp import Llama; import os; llm = Llama(model_path=os.path.expanduser('~/models/your-model.gguf'), n_gpu_layers=35); print(llm('Hello', max_tokens=128)['choices'][0]['text'])\""
echo ""
echo "3. Use interactive Python script:"
echo "   python3 ~/example_llama_inference.py"
echo ""
echo "4. Check GPU usage:"
echo "   watch -n 1 nvidia-smi"
echo ""
echo -e "${GREEN}Setup complete!${NC}"

