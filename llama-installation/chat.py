#!/usr/bin/env python3
"""
Simple interactive chat script for Llama model on Jetson
Usage: python3 chat.py
"""

from llama_cpp import Llama
import os

# Load model
llm = Llama(
    model_path=os.path.expanduser('~/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf'),
    n_gpu_layers=35
)

print("Chat started! Type 'quit' to exit.\n")

while True:
    prompt = input("You: ")
    if prompt.lower() in ['quit', 'exit', 'q']:
        break
    
    response = llm(prompt, max_tokens=100)
    print("AI:", response['choices'][0]['text'], "\n")

