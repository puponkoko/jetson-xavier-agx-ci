#!/usr/bin/env python3
"""
Example script for running Llama inference on Jetson Xavier AGX
Usage: python3 example_llama_inference.py
"""

from llama_cpp import Llama
import sys
import os

# Configuration - defaults to TinyLlama (most commonly downloaded)
MODEL_PATH = os.path.expanduser("~/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
# Alternative paths to try:
# MODEL_PATH = os.path.expanduser("~/models/llama-3.1-minitron-4b-width-base-gguf/model.gguf")
# MODEL_PATH = os.path.expanduser("~/models/llama-2-7b-chat.Q4_K_M.gguf")

# Find any .gguf file in models directory if default doesn't exist
if not os.path.exists(MODEL_PATH):
    models_dir = os.path.expanduser("~/models")
    if os.path.exists(models_dir):
        for root, dirs, files in os.walk(models_dir):
            for file in files:
                if file.endswith('.gguf'):
                    MODEL_PATH = os.path.join(root, file)
                    print(f"Found model: {MODEL_PATH}")
                    break
            if MODEL_PATH and os.path.exists(MODEL_PATH):
                break

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model not found at {MODEL_PATH}")
    print("Please download a model first:")
    print("  cd ~/models")
    print("  hf download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --local-dir .")
    print("  # Or use: huggingface-cli download ... (if hf command not available)")
    sys.exit(1)

print(f"Loading model: {MODEL_PATH}")
print("This may take a minute...")

# Initialize Llama with GPU offloading
# n_gpu_layers: Number of layers to offload to GPU (higher = more GPU memory, faster)
# Adjust based on your model size and available GPU memory
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,        # Context window size
    n_threads=8,       # Number of CPU threads (adjust based on your CPU)
    n_gpu_layers=35,   # Layers to offload to GPU (0 = CPU only, higher = more GPU)
    verbose=True
)

print("Model loaded! Ready for inference.")
print("Type 'quit' or 'exit' to stop.\n")

# Interactive loop
while True:
    try:
        prompt = input("You: ")
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not prompt.strip():
            continue
        
        print("\nAssistant: ", end="", flush=True)
        
        # Generate response
        response = llm(
            prompt,
            max_tokens=256,      # Maximum tokens to generate
            temperature=0.7,    # Randomness (0.0 = deterministic, 1.0 = creative)
            top_p=0.9,          # Nucleus sampling
            repeat_penalty=1.1, # Penalty for repetition
            stream=True         # Stream output token by token
        )
        
        # Stream the response
        full_response = ""
        for output in response:
            text = output['choices'][0]['text']
            print(text, end="", flush=True)
            full_response += text
        
        print("\n")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
        break
    except Exception as e:
        print(f"\nError: {e}")
        break

